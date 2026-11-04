#include "reference_config.h"
#include "ds4-adapter.c"
#include "manifest.h"
#include <sys/sysctl.h>
extern void dequantize_row_q8_0(const void*,float*,int64_t);
extern void dequantize_row_q5_K(const void*,float*,int64_t);
extern void dequantize_row_iq4_nl(const void*,float*,int64_t);
#define CHECK(x) do {if(!(x)){fprintf(stderr,"reference failure at line %d: %s\n",__LINE__,#x);exit(1);}}while(0)
static const char *capture;
static bool logits_only;
static double reference_seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static unsigned argmax(const float *logits) {
    unsigned best=0;
    // Bit inspection survives the upstream host build's fast-math flags.
    for(unsigned i=0;i<129280;i++) {
        uint32_t bits;memcpy(&bits,logits+i,4);CHECK((bits&0x7f800000u)!=0x7f800000u);
        if(logits[i]>logits[best])best=i;
    }
    return best;
}

static void record(ds4_gpu_tensor *t,uint32_t pos,int layer,const char *name,size_t count) {
    if(!capture || (logits_only && strcmp(name,"logits")))return;
    bool active=ds4_gpu_commands_active();if(active)CHECK(ds4_gpu_end_commands());
    char path[1024];if(layer<0)snprintf(path,sizeof(path),"%s/p%u.%s.bin",capture,pos,name);
    else snprintf(path,sizeof(path),"%s/p%u.l%d.%s.bin",capture,pos,layer,name);
    float *v=malloc(count*4);CHECK(v);CHECK(ds4_gpu_tensor_read(t,0,v,count*4));
    FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(v,4,count,f)==count);fclose(f);free(v);if(active)CHECK(ds4_gpu_begin_commands());
}
int main(int argc,char **argv) {
    CHECK(argc>=3);
    bool benchmark=!strcmp(argv[1],"--benchmark");
    logits_only=!strcmp(argv[1],"--logits");
    unsigned generated_count=benchmark?(unsigned)strtoul(argv[2],NULL,10):0;
    unsigned first_token=(benchmark||logits_only)?3:2;
    CHECK((unsigned)argc>first_token);if(benchmark)CHECK(generated_count>0&&generated_count<=256);
    capture=benchmark?NULL:strcmp(argv[logits_only?2:1],"-")?argv[logits_only?2:1]:NULL;
    if(capture)CHECK(mkdir(capture,0700)==0);
    g_ds4_shape=DS4_SHAPE_FLASH41;g_ds4_shape.n_expert=192;
    uint32_t ctx=(uint32_t)argc-first_token+generated_count+1;if(ctx<256)ctx=256;
    CHECK(ctx<=1048576);uint64_t phys;size_t sz=sizeof(phys);CHECK(!sysctlbyname("hw.memsize",&phys,&sz,NULL,0));
    CHECK(phys>(8ull<<30));uint64_t cap=phys-(8ull<<30);
    const char *limit=getenv("LISA_RAM_CAP_GB");
    if(limit){char *end=NULL;double gib=strtod(limit,&end);CHECK(end!=limit&&!*end&&gib>0&&gib<=(double)cap/(1ull<<30));cap=(uint64_t)(gib*(1ull<<30));}
    CHECK(resident_bytes+ds41_graph_bytes(ctx)<cap);
    for(unsigned i=0;i<40;i++)g_ds4_compress_ratios[i]=ds4_expected_layer_compress_ratio(i);
    uint8_t *map=mmap(NULL,resident_bytes,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);CHECK(map!=MAP_FAILED);
    for(unsigned i=0;i<sizeof(shards)/sizeof(shards[0]);i++) {
        int fd=open(shards[i].path,O_RDONLY);CHECK(fd>=0);
        CHECK(mmap(map+shards[i].base,shards[i].size,PROT_READ,MAP_SHARED|MAP_FIXED,fd,0)==map+shards[i].base);close(fd);
    }
    ds4_model model={.map=map,.size=resident_bytes,.tensors=tensors,.n_tensors=sizeof(tensors)/sizeof(tensors[0])};
    ds4_weights weights={0};bind_weights(&model,&weights);
    CHECK(ds4_gpu_init());
    uint64_t offsets[5],sizes[5],largest=0;
    for(unsigned i=0;i<model.n_tensors;i++)if(tensors[i].bytes>largest)largest=tensors[i].bytes;
    for(unsigned i=0;i<5;i++){offsets[i]=shards[i].base;sizes[i]=shards[i].size;}
    CHECK(ds4_gpu_set_model_map_spans(map,resident_bytes,offsets,sizes,5,largest));
    ds41_gpu_graph graph={0},*g=&graph;g->ctx=ctx;g->prefill_cap=1;g->tp_world=1;g->valid=true;g->scalar_decode=true;g->index_bf16_ok=true;
#define ALLOC(name,count) CHECK((g->name=ds4_gpu_tensor_alloc((uint64_t)((count)?(count):1)*4)));
    DS41_SCRATCH(ALLOC)
#undef ALLOC
    CHECK((g->rope_tab=ds4_gpu_tensor_alloc(64*2*4)));
    for(unsigned i=0;i<40;i++)CHECK((g->window[i]=ds4_gpu_tensor_alloc(128*512*4)));
    for(unsigned i=0;i<4;i++) {
        unsigned cap=ctx/(i<3?2:1)+1;
        CHECK((g->compressed[i]=ds4_gpu_tensor_alloc((uint64_t)cap*512*4)));
        CHECK((g->index_cache[i]=ds4_gpu_tensor_alloc((uint64_t)cap*128*4)));
        CHECK((g->index_bf16[i]=ds4_gpu_tensor_alloc((uint64_t)cap*128*2)));
        CHECK((g->previous_kv[i]=ds4_gpu_tensor_alloc(512*4)));
        CHECK((g->previous_score[i]=ds4_gpu_tensor_alloc(512*4)));
    }
    CHECK((g->index_bf16_flag=ds4_gpu_tensor_alloc(4)));
    uint32_t *token_map=malloc(129280*4);FILE *mf=fopen(REFERENCE_TOKEN_MAP,"rb");CHECK(mf);CHECK(fread(token_map,4,129280,mf)==129280);fclose(mf);
    engram.token_map=token_map;CHECK(ds4_engram_layout_valid(&engram));ds4_engram_history history;ds4_engram_history_reset(&history);
    float ones[20480];for(unsigned i=0;i<20480;i++)ones[i]=1;
    for(unsigned i=0;i<2;i++) {
        ds4_tensor *w=required_tensorf(&model,"blk.%u.engram_gate.weight",i?14:1);
        CHECK((g->engram_q_norm[i]=ds4_gpu_tensor_alloc(20480*4)));CHECK(ds4_gpu_tensor_write(g->engram_q_norm[i],0,map+w->abs_offset,20480*4));
        CHECK((g->engram_k_norm[i]=ds4_gpu_tensor_alloc(20480*4)));CHECK(ds4_gpu_tensor_write(g->engram_k_norm[i],0,ones,20480*4));
    }
    int fd=open(engram_path,O_RDONLY);CHECK(fd>=0);CHECK(!fcntl(fd,F_NOCACHE,1));CHECK(!fcntl(fd,F_RDAHEAD,0));
    g->engram=engram;ds4_engram_history_reset(&g->history);
    for(unsigned i=0;i<2;i++)CHECK(ds4_engram_table_open(&g->table[i],engram_path,
        engram_offset+(i?(uint64_t)engram.rows[0]*144:0),engram.rows[i]));
    CHECK((g->engram_reader=ds4_engram_reader_create()));
    if(benchmark || logits_only) {
        float logits[129280];unsigned best=0;
        const double begin=reference_seconds();
        for(unsigned i=first_token;i<(unsigned)argc;i++){
            int token=atoi(argv[i]);CHECK(token>=0&&token<129280);
            CHECK(ds41_graph_step(g,&model,&weights,token,logits));
            record(g->logits,g->pos-1,-1,"logits",129280);
        }
        const double prefill=reference_seconds()-begin;
        best=argmax(logits);
        const double decode_begin=reference_seconds();
        if(benchmark) {
            printf("{\"generated_ids\":[");
            for(unsigned n=0;n<generated_count;n++) {
                if(n)printf(",");printf("%u",best);
                if(n+1==generated_count)break;
                CHECK(ds41_graph_step(g,&model,&weights,(int)best,logits));
                best=0;best=argmax(logits);
            }
            double elapsed=reference_seconds()-decode_begin;
            printf("],\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f,\"decode_steps\":%u,\"decode_tokens_per_second\":%.9f}\n",
                prefill,elapsed,generated_count-1,(generated_count-1)/elapsed);
        }
        ds4_engram_reader_free(g->engram_reader);g->engram_reader=NULL;
        CHECK(ds4_gpu_synchronize());return 0;
    }
    for(unsigned pos=0;pos<(unsigned)argc-first_token;pos++) {
        int token=atoi(argv[pos+first_token]);CHECK(token>=0&&token<129280);g->pos=pos;
        uint32_t ids[2][24];CHECK(ds4_engram_hash(&engram,&history,&token,NULL,1,&ids[0][0]));
        for(unsigned layer=0;layer<2;layer++)for(unsigned h=0;h<24;h++) {
            uint8_t packed[144];uint64_t row=ids[layer][h]+(layer?engram.rows[0]:0ull);
            CHECK(pread(fd,packed,144,engram_offset+row*144)==144);
            dequantize_row_iq4_nl(packed,g->rows[layer]+h*256,256);
        }
        CHECK(ds4_gpu_tensor_write(g->engram_rows,0,g->rows[0],24*256*4));
        CHECK(ds4_gpu_tensor_write(g->engram_rows_b,0,g->rows[1],24*256*4));
        float embedding[20480];ds4_tensor *w=weights.token_embd;uint64_t row=w->bytes/129280;
        if(w->type==8)dequantize_row_q8_0(map+w->abs_offset+row*token,embedding,5120);
        else {CHECK(w->type==13);dequantize_row_q5_K(map+w->abs_offset+row*token,embedding,5120);}
        for(unsigned i=1;i<4;i++)memcpy(embedding+i*5120,embedding,5120*4);
        const float pre[4]={1,0,0,0};CHECK(ds4_gpu_tensor_write(g->residual,0,embedding,sizeof(embedding)));CHECK(ds4_gpu_tensor_write(g->pre,0,pre,sizeof(pre)));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_rope_coeff_table(g->rope_tab,pos));g->rope_tab_pos=pos+1;
        for(unsigned l=0;l<40;l++) {
            const ds4_layer_weights *lw=&weights.layer[l];
            CHECK(ds41_graph_before_attention(g,&model,lw,l));
            if(ds41_engram_layer(l))record(g->residual,pos,l,"engram",20480);
            record(g->norm,pos,l,"attention.norm",5120);record(g->attn_split,pos,l,"attention.split",24);
            CHECK(ds41_attention(g,&model,lw,l,false));
            record(g->block,pos,l,"attention.output",5120);
            record(g->q,pos,l,"attention.query",64*512);ds4_gpu_tensor *rv=ds4_gpu_tensor_view(g->window[l],(pos%128)*512ull*4,512*4);record(rv,pos,l,"attention.raw",512);ds4_gpu_tensor_free(rv);
            if(ds41_kv_source(l)) {record(g->pool_kv,pos,l,"attention.pool",512);record(g->pool_score,pos,l,"attention.pool_score",512);
                if((pos+1)%ds4_layer_compress_ratio(l)==0){unsigned own=l<8?0:l<14?1:l<20?2:3; unsigned nr=(pos+1)/ds4_layer_compress_ratio(l)-1; ds4_gpu_tensor *cv=ds4_gpu_tensor_view(g->compressed[own],nr*512ull*4,512*4);record(cv,pos,l,"attention.latent",512);ds4_gpu_tensor_free(cv); cv=ds4_gpu_tensor_view(g->index_cache[own],nr*128ull*4,128*4);record(cv,pos,l,"attention.index_key",128);ds4_gpu_tensor_free(cv); record(g->latent,pos,l,"attention.latent_norm",512);}}
            if(ds41_index_source(l)&&((pos+1)/ds4_layer_compress_ratio(l)))record(g->index_q,pos,l,"attention.index_query",32*128);

            CHECK(ds41_hc_mix(g,&model,lw,true,g->attn_split));
            record(g->norm,pos,l,"ffn.norm",5120);record(g->ffn_split,pos,l,"ffn.split",24);
            CHECK(ds41_moe(g,&model,lw,l,token));
            record(g->selected,pos,l,"ffn.router",6);record(g->route_weights,pos,l,"ffn.weights",6);record(g->routed,pos,l,"ffn.routed",5120);
            CHECK(ds41_graph_after_moe(g,&model,lw,l,l==39));
            record(g->shared,pos,l,"ffn.shared",5120);record(g->residual,pos,l,"residual",20480);
        }
        CHECK(ds41_graph_encode_logits(g,&model,&weights));CHECK(ds4_gpu_end_commands());
        record(g->norm,pos,-1,"hidden",5120);record(g->logits,pos,-1,"logits",129280);
        float logits[129280];CHECK(ds4_gpu_tensor_read(g->logits,0,logits,sizeof(logits)));unsigned best=0;best=argmax(logits);
        printf("position %u token %d next %u\n",pos,token,best);fflush(stdout);
    }
    CHECK(ds4_gpu_synchronize());return 0;
}
