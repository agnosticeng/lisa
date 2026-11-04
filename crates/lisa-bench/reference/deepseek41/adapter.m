#include "reference_config.h"
#include "ds4_metal.m"
static id<MTLLibrary> adapter_library;
static NSMutableDictionary *adapter_pipelines;
static id<MTLComputePipelineState> adapter_pipeline(uint32_t type, bool indirect) {
    @autoreleasepool {
        if (!adapter_library) {
            NSError *error=nil;
            NSString *source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:REFERENCE_PACKED] encoding:NSUTF8StringEncoding error:&error];
            MTLCompileOptions *options=[MTLCompileOptions new];
            adapter_library=[g_device newLibraryWithSource:source options:options error:&error];
            if (!adapter_library) {fprintf(stderr,"adapter compile: %s\n",error.description.UTF8String);return nil;}
            adapter_pipelines=[NSMutableDictionary new];
        }
        NSString *format=type==13?@"q5_K":type==29?@"iq1_m":type==17?@"iq2_xs":@"iq2_xxs";
        NSString *name=[NSString stringWithFormat:@"lisa_gguf_%@_%@",format,indirect?@"id":@"mv"];
        id<MTLComputePipelineState> pipeline=adapter_pipelines[name];
        if (!pipeline) {
            MTLFunctionConstantValues *constants=[MTLFunctionConstantValues new];
            short two=2,one=1;bool no=false;
            [constants setConstantValue:&two type:MTLDataTypeShort atIndex:600];
            for(unsigned i=602;i<=604;i++)[constants setConstantValue:&one type:MTLDataTypeShort atIndex:i];
            [constants setConstantValue:&no type:MTLDataTypeBool atIndex:605];
            NSError *error=nil;
            id<MTLFunction> fn=[adapter_library newFunctionWithName:name constantValues:constants error:&error];
            pipeline=[g_device newComputePipelineStateWithFunction:fn error:&error];
            if(!pipeline){fprintf(stderr,"adapter pipeline: %s\n",error.description.UTF8String);return nil;}
            adapter_pipelines[name]=pipeline;
        }
        return pipeline;
    }
}
int adapter_mv(ds4_gpu_tensor *out,const void *map,uint64_t size,uint64_t offset,uint32_t type,uint32_t k,uint32_t n,const ds4_gpu_tensor *x) {
    if (type==30) {
        static const void *keys[128];static ds4_gpu_tensor *views[128];static unsigned count;
        unsigned i=0;for(;i<count;i++)if(keys[i]==(const char*)map+offset)break;
        if(i==count){if(count==128)return 0;keys[i]=(const char*)map+offset;views[i]=ds4_gpu_tensor_alloc((uint64_t)k*n*2);if(!views[i]||!ds4_gpu_tensor_write(views[i],0,keys[i],(uint64_t)k*n*2))return 0;count++;}
        return ds4_gpu_matmul_bf16_weights_tensor(out,views[i],k,n,x);
    }
    uint64_t row=(uint64_t)k/256*176,inner=0;
    id<MTLBuffer> w=ds4_gpu_wrap_model_range(map,size,offset,row*n,&inner);
    id<MTLComputePipelineState> pipeline=adapter_pipeline(type,false);if(!w||!pipeline)return 0;
    ds4_gpu_q8_0_matvec_args a={.ne00=k,.ne01=n,.ne02=1,.nb00=176,.nb01=row,.nb02=row*n,.nb03=row*n,
        .ne10=k,.ne11=1,.ne12=1,.nb10=4,.nb11=k*4ull,.nb12=k*4ull,.nb13=k*4ull,.ne0=n,.ne1=1,.nr0=4,.r2=1,.r3=1};
    int owned=0;id<MTLCommandBuffer> cb=ds4_gpu_command_buffer(&owned);id<MTLComputeCommandEncoder> enc=ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];[enc setBytes:&a length:sizeof(a) atIndex:0];[enc setBuffer:w offset:inner atIndex:1];
    [enc setBuffer:ds4_gpu_tensor_buffer(x) offset:ds4_gpu_tensor_offset(x) atIndex:2];[enc setBuffer:ds4_gpu_tensor_buffer(out) offset:ds4_gpu_tensor_offset(out) atIndex:3];
    [enc setThreadgroupMemoryLength:16 atIndex:0];[enc dispatchThreadgroups:MTLSizeMake(n/8,1,1) threadsPerThreadgroup:MTLSizeMake(32,2,1)];
    ds4_gpu_end_compute_encoder(cb,enc);return ds4_gpu_finish_command_buffer(cb,owned,"GGML packed adapter");
}
int adapter_expert(ds4_gpu_tensor *out,const void *map,uint64_t size,uint64_t offset,uint32_t type,uint32_t k,uint32_t n,const ds4_gpu_tensor *x,const ds4_gpu_tensor *ids,uint32_t slots) {
    uint32_t block=type==29?56:type==17?74:66;uint64_t row=(uint64_t)k/256*block,inner=0;
    id<MTLBuffer> w=ds4_gpu_wrap_model_range(map,size,offset,row*n*192,&inner);
    id<MTLComputePipelineState> pipeline=adapter_pipeline(type,true);if(!w||!pipeline)return 0;
    ds4_gpu_mul_mv_id_args a={.nei0=6,.nei1=1,.nbi1=24,.ne00=k,.ne01=n,.ne02=192,.nb00=block,.nb01=row,.nb02=row*n,
        .ne10=k,.ne11=slots,.ne12=1,.ne13=1,.nb10=4,.nb11=k*4ull,.nb12=k*slots*4ull,.ne0=n,.ne1=6,.nb1=n*4ull,.nr0=4,.tp_world=1};
    int owned=0;id<MTLCommandBuffer> cb=ds4_gpu_command_buffer(&owned);id<MTLComputeCommandEncoder> enc=ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];[enc setBytes:&a length:sizeof(a) atIndex:0];[enc setBuffer:w offset:inner atIndex:1];
    [enc setBuffer:ds4_gpu_tensor_buffer(x) offset:ds4_gpu_tensor_offset(x) atIndex:2];[enc setBuffer:ds4_gpu_tensor_buffer(out) offset:ds4_gpu_tensor_offset(out) atIndex:3];
    [enc setBuffer:ds4_gpu_tensor_buffer(ids) offset:ds4_gpu_tensor_offset(ids) atIndex:4];
    [enc setThreadgroupMemoryLength:type==17?4224:type==16?2176:16 atIndex:0];[enc dispatchThreadgroups:MTLSizeMake(n/8,1,6) threadsPerThreadgroup:MTLSizeMake(32,2,1)];
    ds4_gpu_end_compute_encoder(cb,enc);return ds4_gpu_finish_command_buffer(cb,owned,"GGML expert adapter");
}
int adapter_activation(ds4_gpu_tensor *mid,ds4_gpu_tensor *gate,ds4_gpu_tensor *up,ds4_gpu_tensor *weights) {
    int owned=0;id<MTLCommandBuffer> cb=ds4_gpu_command_buffer(&owned);
    int ok=ds4_gpu_encode_moe_swiglu_weight(cb,ds4_gpu_tensor_buffer(gate),ds4_gpu_tensor_offset(gate),ds4_gpu_tensor_buffer(up),ds4_gpu_tensor_offset(up),ds4_gpu_tensor_buffer(mid),ds4_gpu_tensor_offset(mid),ds4_gpu_tensor_buffer(weights),ds4_gpu_tensor_offset(weights),2304,6,10,false);
    return ok&&ds4_gpu_finish_command_buffer(cb,owned,"DS4 routed activation");
}
int adapter_sum(ds4_gpu_tensor *out,ds4_gpu_tensor *experts) {
    int owned=0;id<MTLCommandBuffer> cb=ds4_gpu_command_buffer(&owned);
    int ok=ds4_gpu_encode_moe_sum6(cb,ds4_gpu_tensor_buffer(experts),ds4_gpu_tensor_offset(experts),ds4_gpu_tensor_buffer(out),ds4_gpu_tensor_offset(out),5120,1);
    return ok&&ds4_gpu_finish_command_buffer(cb,owned,"DS4 routed reduction");
}
