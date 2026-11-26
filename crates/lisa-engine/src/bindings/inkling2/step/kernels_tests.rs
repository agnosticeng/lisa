use super::*;

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn float(bytes: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn sized<T: Abi>(p: &T) -> Vec<u8> {
    let bytes = p.bytes();
    assert_eq!(bytes.len(), T::SIZE);
    bytes
}

#[test]
fn vendored_block_residual_mix_params() {
    let b = sized(&BlockResidualMixParams {
        rows: 3,
        hidden: 2304,
        stack: 2,
        queries: 4,
        stack_plane: 16 * 2304,
        scale: 0.5,
    });
    assert_eq!(BlockResidualMixParams::SIZE, 24);
    assert_eq!((word(&b, 0), word(&b, 8), word(&b, 16)), (3, 2, 16 * 2304));
    assert_eq!(float(&b, 20), 0.5);
}

#[test]
fn vendored_message_gate_params() {
    let b = sized(&MessageGateFusedParams {
        rows: 5,
        hidden: 2304,
        x_stride: 6912,
        message_stride: 9216,
        rank: 64,
        gates: 3,
        x_columns: [0, 2304, 4608],
        message_columns: [11, 12, 13],
    });
    assert_eq!(MessageGateFusedParams::SIZE, 48);
    assert_eq!((word(&b, 16), word(&b, 20)), (64, 3));
    assert_eq!((word(&b, 28), word(&b, 36), word(&b, 44)), (2304, 11, 13));
}

#[test]
fn vendored_short_convolution_params() {
    let p = ShortConvolutionParams {
        rows: 7,
        output: ConvolutionOutput::RoundedSum,
        tree: 1,
        state_column: 99,
        divisor: 5f32.sqrt(),
        dynamic_stride: 64,
        capture_sequences: [1, 2, 3, 4],
        capture_rows: [5, 6, 7, 8],
        input_stride: 2048,
        pair_state_column: 1234,
        ..Default::default()
    };
    let b = sized(&p);
    assert_eq!(ShortConvolutionParams::SIZE, 104);
    assert_eq!(
        (word(&b, 0), word(&b, 12), word(&b, 24), word(&b, 36)),
        (7, 2, 1, 99)
    );
    assert_eq!(float(&b, 40), 5f32.sqrt());
    assert_eq!((word(&b, 52), word(&b, 56), word(&b, 72)), (64, 1, 5));
    assert_eq!((word(&b, 88), word(&b, 100)), (2048, 1234));
    let c = sized(&MoeCombineConvolutionParams {
        combine: MoeCombineParams {
            rows: 7,
            hidden_size: 2304,
            routes_per_row: 8,
        },
        convolution: p,
    });
    assert_eq!(MoeCombineConvolutionParams::SIZE, 116);
    assert_eq!((word(&c, 8), word(&c, 12), word(&c, 12 + 36)), (8, 7, 99));
    assert_eq!(&c[12..], &b[..]);
}

#[test]
fn vendored_attention_params() {
    let mut p = AttentionParams {
        sequences: 2,
        rows: 12,
        trees: 1,
        kv: KvLayer {
            extent_pages: 64,
            offset: 4096,
        },
        ..Default::default()
    };
    p.sequence[1] = attention_sequence(1, 100, 8, 5, 272);
    p.sequence[7].block_list = 77;
    let b = sized(&p);
    assert_eq!(AttentionParams::SIZE, 384);
    assert_eq!(AttentionSequence::SIZE, 44);
    assert_eq!((word(&b, 0), word(&b, 8), word(&b, 16)), (2, 12, 1));
    assert_eq!((word(&b, 24), word(&b, 28)), (64, 4096));
    let s1 = 32 + 44;
    assert_eq!(word(&b, s1), 100);
    assert_eq!(word(&b, s1 + 12), 8 * 32);
    assert_eq!(word(&b, s1 + 20), 4);
    assert_eq!(
        (word(&b, s1 + 24), word(&b, s1 + 28), word(&b, s1 + 32)),
        (5, 1, 272)
    );
    assert_eq!(word(&b, 32 + 7 * 44 + 40), 77);
}

#[test]
fn vendored_path_commit_and_moe_params() {
    let mut commit = KvPathCommitParams {
        extent_pages: 64,
        regions: 48,
        ..Default::default()
    };
    commit.committed_tokens[7] = 9;
    let b = sized(&commit);
    assert_eq!(KvPathCommitParams::SIZE, 40);
    assert_eq!((word(&b, 4), word(&b, 36)), (48, 9));

    let b = sized(&MoeBiasedRouteParams {
        route: MoeRouteParams {
            rows: 3,
            input_size: 2304,
            experts: 128,
            top_k: 8,
        },
        scale: 8.0,
    });
    assert_eq!((MoeRouteParams::SIZE, MoeBiasedRouteParams::SIZE), (16, 20));
    assert_eq!((word(&b, 8), word(&b, 12), float(&b, 16)), (128, 8, 8.0));

    let b = sized(&MoeGroupParams {
        rows: 3,
        top_k: 8,
        tile_rows: 8,
        experts: 128,
        gate_up_columns: 36,
        down_columns: 36,
        streamed: 0,
        lanes: 1,
    });
    assert_eq!(MoeGroupParams::SIZE, 32);
    assert_eq!((word(&b, 16), word(&b, 28)), (36, 1));

    assert_eq!(
        sized(&MoeGatherParams {
            input_size: 2304,
            routes_per_row: 8
        })
        .len(),
        8
    );

    let b = sized(&MoeGgufGateUpParams {
        input_size: 2304,
        output_size: 1152,
        up_routed_format: 3,
        chunk_slots: 128,
        tiles: 2,
        ..Default::default()
    });
    assert_eq!(MoeGgufGateUpParams::SIZE, 40);
    assert_eq!(
        (word(&b, 4), word(&b, 20), word(&b, 32), word(&b, 36)),
        (1152, 3, 128, 2)
    );

    let b = sized(&MoeGgufExpertParams {
        routed_format: 4,
        tiles: 1,
        ..Default::default()
    });
    assert_eq!(MoeGgufExpertParams::SIZE, 32);
    assert_eq!((word(&b, 12), word(&b, 28)), (4, 1));
}

#[test]
fn vendored_gguf_params() {
    let b = sized(&GgufFloatParams {
        rows: 3,
        input_size: 2304,
        output_size: [24, 0],
        out_stride: [6592, 0],
        out_offset: [6528, 0],
    });
    assert_eq!(GgufFloatParams::SIZE, 32);
    assert_eq!((word(&b, 8), word(&b, 16), word(&b, 24)), (24, 6592, 6528));

    assert_eq!(
        sized(&GgufEmbedParams {
            rows: 1,
            vocabulary: 2,
            hidden: 3
        })
        .len(),
        12
    );

    let b = sized(&RowCopyParams {
        destination_stride: 9216,
        destination_column: 4608,
        ..Default::default()
    });
    assert_eq!(RowCopyParams::SIZE, 32);
    assert_eq!((word(&b, 24), word(&b, 28)), (9216, 4608));

    let b = sized(&LogitFinishParams {
        rows: 3,
        vocabulary: 201024,
        tokens: 200058,
        softcap: 30.0,
    });
    assert_eq!(LogitFinishParams::SIZE, 16);
    assert_eq!((word(&b, 8), float(&b, 12)), (200058, 30.0));

    let b = sized(&GgufColumnParams {
        rows: 2,
        format: Q80,
        ..Default::default()
    });
    assert_eq!(GgufColumnParams::SIZE, 20);
    assert_eq!(word(&b, 16), Q80);

    let kv = [
        Segment {
            columns: 1024,
            format: Q80,
            offset: 0,
        },
        Segment {
            columns: 1024,
            format: Q80,
            offset: 1024,
        },
    ];
    let b = sized(&GgufColumnFusedParams::new(2, 2304, 2048, &kv).unwrap());
    assert_eq!(GgufColumnFusedParams::SIZE, 48);
    assert_eq!((word(&b, 12), word(&b, 16), word(&b, 20)), (1024, 1024, 0));
    assert_eq!((word(&b, 24), word(&b, 32), word(&b, 40)), (Q80, 0, 1024));

    let b = sized(&GgufDecodeParams {
        input_size: 3072,
        splits: 8,
        out_stride: 2304,
        spread: 1,
        ..Default::default()
    });
    assert_eq!(GgufDecodeParams::SIZE, 20);
    assert_eq!((word(&b, 4), word(&b, 16)), (8, 1));

    let b = sized(&GgufDecodeFusedParams::new(2304, 2, 6592, &kv, 1).unwrap());
    assert_eq!(GgufDecodeFusedParams::SIZE, 52);
    assert_eq!((word(&b, 16), word(&b, 40), word(&b, 48)), (1024, 1024, 1));
    assert!(GgufDecodeFusedParams::new(2304, 2, 6592, &[], 1).is_err());

    assert_eq!(width_bytes(2304), 2304u32.to_le_bytes());
}

#[test]
fn vendored_sampling_params() {
    let p = TargetSamplingParams::greedy(201024, &[200006]).unwrap();
    let b = sized(&p);
    assert_eq!(TargetSamplingParams::SIZE, 172);
    assert_eq!((word(&b, 0), word(&b, 4)), (201024, 6282));
    assert_eq!(
        (word(&b, 152), word(&b, 156), word(&b, 164)),
        (200006, NO_TOKEN, NO_TOKEN)
    );
    assert_eq!(word(&b, 168), 1);

    let mut accept = AcceptBatchParams::default();
    accept.remaining[0] = 16;
    accept.stop_tokens[0] = 200006;
    let b = sized(&accept);
    assert_eq!(AcceptBatchParams::SIZE, 48);
    assert_eq!(
        (word(&b, 0), word(&b, 32), word(&b, 44)),
        (16, 200006, NO_TOKEN)
    );
}

#[test]
fn vendored_tree_and_lane_rows() {
    let chain = TreeRow::chain(3).unwrap();
    assert_eq!(
        chain[0],
        TreeRow {
            ancestors: 1,
            parent: TREE_ROOT,
            depth: 0
        }
    );
    assert_eq!(
        chain[2],
        TreeRow {
            ancestors: 7,
            parent: 1,
            depth: 2
        }
    );
    assert_eq!(TreeRow::chain(32).unwrap()[31].ancestors, u32::MAX);
    let b = pack(&chain);
    assert_eq!(b.len(), 24);
    assert_eq!(word(&b, 16), 7);
    assert_eq!(u16::from_le_bytes([b[20], b[21]]), 1);
    assert_eq!(u16::from_le_bytes([b[22], b[23]]), 2);
    assert_eq!(&b[4..6], &[0xff, 0xff]);
    let lanes = pack(&[
        LaneRows { first: 0, rows: 5 },
        LaneRows { first: 5, rows: 3 },
    ]);
    assert_eq!((word(&lanes, 4), word(&lanes, 8)), (5, 5));
}

#[test]
fn vendored_pass_formats() {
    assert_eq!(pass_formats::<4>(&[4, 0, 4, 0]).unwrap(), [0, 4, 4, 4]);
    assert_eq!(pass_formats::<4>(&[0, 0, 0, 0]).unwrap(), [0; 4]);
    assert_eq!(pass_formats::<2>(&[6, 6]).unwrap(), [6, 6]);
    assert_eq!(pass_formats::<3>(&[Q80, 3]).unwrap(), [3, Q80, Q80]);
    assert!(pass_formats::<2>(&[1, 2, 3]).is_err());
}

#[test]
fn vendored_decode_splits_on_forty_cores() {
    assert_eq!(
        (decode_most_splits(8, 40), decode_most_splits(16, 40)),
        (18, 12)
    );
    assert_eq!(decode_splits(1, 40, 8), 1);
    assert_eq!(decode_splits(5, 40, 8), 5);
    // A sliding layer's window: 34 pages, two a split.
    assert_eq!(window_attention_pages(WINDOW_TOKENS), 34);
    assert_eq!(read_pages(5000, WINDOW_TOKENS), 34);
    assert_eq!(decode_splits(34, 40, 8), 17);
    assert_eq!(decode_splits(40, 40, 8), 14);
    assert_eq!(decode_splits(100, 40, 16), 12);
    assert_eq!(read_pages(100 + 16, 0), 4);
    assert_eq!(decode_splits(4, 40, 16), 4);
    assert_eq!(split_reads(34, 17), 2);
    assert_eq!(page_table_entries(95, 16), 4);
}

#[test]
fn vendored_attention_workspace() {
    assert_eq!(attention_first_slots(&[17, 5], 8), [0, 272]);
    assert_eq!(attention_first_slots(&[3, 2], 16), [0, 96]);
    assert_eq!(
        attention_workspace_bytes(&[17, 5], 8),
        (44 * 192 * 128 * 4, 44 * 192 * 2 * 4)
    );
    let lane = AttentionSlots {
        first_slot: 272,
        splits: 5,
        rows_per_tile: 8,
    };
    assert_eq!(lane.fused_rows(), 24);
    // Head slot 272 + 2 * 5; split 3.
    assert_eq!(lane.partials(2, 3), (272 + 10 + 3) * 24 * 128);
    assert_eq!(lane.statistics(2, 3, 7), ((272 + 10 + 3) * 24 + 7) * 2);
    assert_eq!(lane.staging(2, 3), (272 + 10 * 5 + 3) * 24 * 128);
    // A 16-row pipeline's slots are two tiles: its first slot counts tiles.
    let wide = AttentionSlots {
        first_slot: 96,
        splits: 2,
        rows_per_tile: 16,
    };
    assert_eq!(wide.partials(0, 1), (48 + 1) * 48 * 128);
    // Its last staging slot ends inside the workspace of both lanes.
    let (partials, _) = attention_workspace_bytes(&[3, 2], 16);
    assert_eq!((wide.staging(7, 1) + 48 * 128) * 4, partials);
    assert_eq!(attention_counter(0, 1, 3), 11);
    let layer = kv_layer(64, 3).unwrap();
    assert_eq!(layer.offset, 3 * 64 * 2 * (32768 + 1024));
    assert!(kv_layer(1024, 64).is_err());
    assert_eq!(kv_extent_bytes(64, 48), 64 * 48 * 67584);
}

#[test]
fn vendored_attention_dispatches() {
    let variant = AttentionVariant {
        int8_queries: false,
        window: true,
        table16: false,
    };
    assert_eq!(
        relative_attention_decode_entry(variant, 8),
        "relative_attention_q8_decode_w1023"
    );
    let full = AttentionVariant {
        int8_queries: true,
        window: false,
        table16: true,
    };
    assert_eq!(
        relative_attention_decode_entry(full, 16),
        "relative_attention_q8q_decode_r16_scaled_table16"
    );
    let mut p = AttentionParams {
        sequences: 2,
        rows: 9,
        trees: 1,
        ..Default::default()
    };
    let firsts = attention_first_slots(&[17, 5], 8);
    p.sequence[0] = attention_sequence(0, 2000, 8, 17, firsts[0]);
    p.sequence[1] = attention_sequence(1, 40, 8, 5, firsts[1]);
    let d = relative_attention_decode(&p, variant, 8).unwrap();
    assert_eq!(d.slots, (0..22).collect::<Vec<_>>());
    assert_eq!((d.bytes_slot, d.bytes.len()), (22, 384));
    assert_eq!((d.grid, d.threads), ([8, 2, 17], [256, 1, 1]));
    p.sequence[1].split_count = 33;
    assert!(relative_attention_decode(&p, variant, 8).is_err());
    let commit = relative_attention_q8_commit(
        &KvPathCommitParams {
            extent_pages: 64,
            regions: 48,
            ..Default::default()
        },
        1,
    )
    .unwrap();
    assert_eq!(
        (commit.bytes_slot, commit.grid, commit.threads),
        (9, [384, 1, 1], [128, 1, 1])
    );
}

#[test]
fn vendored_convolution_and_mix_dispatches() {
    let pair = ShortConvolutionParams {
        rows: 5,
        channels: 1024,
        tree: 1,
        sequences: 1,
        state_stride: 319488,
        ..Default::default()
    };
    let d = short_convolution_pair(&pair, 8).unwrap();
    assert_eq!((d.bytes_slot, d.slots.len(), d.grid), (19, 19, [8, 8, 1]));
    let sum = ShortConvolutionParams {
        channels: 2304,
        output: ConvolutionOutput::Sum,
        ..pair
    };
    let d = short_convolution(&sum, 8).unwrap();
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid),
        ("short_convolution_sum", 28, [8, 9, 1])
    );
    let commit = ShortConvolutionParams {
        channels: 319488,
        ..pair
    };
    let d = short_convolution_commit(&commit).unwrap();
    assert_eq!((d.bytes_slot, d.grid), (19, [3, 1248, 1]));
    let combine = MoeCombineConvolutionParams {
        combine: MoeCombineParams {
            rows: 5,
            hidden_size: 2304,
            routes_per_row: 8,
        },
        convolution: ShortConvolutionParams {
            channels: 2304,
            output: ConvolutionOutput::RoundedSum,
            residual: 1,
            ..pair
        },
    };
    let d = moe_combine_convolution(&combine, 8).unwrap();
    assert_eq!(
        (d.bytes_slot, d.grid, d.threads),
        (18, [72, 1, 1], [32, 8, 1])
    );

    let mix = BlockResidualMixParams {
        rows: 5,
        hidden: 2304,
        stack: 2,
        queries: 3,
        stack_plane: 16 * 2304,
        scale: 1.0,
    };
    let d = block_residual_mix_norm(&mix, MixCurrent::F32).unwrap();
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid, d.threads),
        ("block_residual_mix_norm_f32", 9, [5, 3, 1], [1024, 1, 1])
    );
    assert!(
        block_residual_mix_norm(
            &BlockResidualMixParams { queries: 4, ..mix },
            MixCurrent::Bf16
        )
        .is_err()
    );
    assert_eq!(
        block_residual_mix(
            &BlockResidualMixParams { queries: 4, ..mix },
            MixCurrent::Bf16
        )
        .unwrap()
        .bytes_slot,
        4
    );

    let gate = MessageGateFusedParams {
        rows: 5,
        hidden: 2304,
        x_stride: 3 * 2304,
        message_stride: 9216,
        rank: 64,
        gates: 3,
        x_columns: [0, 2304, 4608],
        message_columns: [0, 2304, 4608],
    };
    assert_eq!(message_gate_normalize(&gate).unwrap().grid, [5, 1, 3]);
    let d = message_gate_down(&gate, GateWeights::Bf16).unwrap();
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid),
        ("message_gate_down_bf16", 5, [32, 1, 3])
    );
    let d = message_gate_apply(&gate, GateWeights::F32).unwrap();
    assert_eq!((d.bytes_slot, d.grid), (18, [9, 1, 3]));
    assert_eq!(
        message_gate_scratch(&gate),
        [3 * 5 * 64 * 2, 3 * 5 * 2304 * 4, 3 * 9 * 8 * 4, 12]
    );
}

#[test]
fn vendored_moe_dispatches() {
    let tiles = moe_maximum_tiles(5, 8, 128, 8);
    assert_eq!(tiles, 40);
    let router = gguf_float(
        true,
        true,
        &GgufFloatParams {
            rows: 5,
            input_size: 2304,
            output_size: [128, 0],
            out_stride: [256, 0],
            ..Default::default()
        },
    );
    assert_eq!(
        (router.entry.as_str(), router.grid),
        ("gguf_float_f32_f32", [16, 1, 1])
    );
    let gate_up = MoeGgufGateUpParams {
        input_size: 2304,
        output_size: 1152,
        experts: 128,
        gate_routed_format: 0,
        gate_shared_format: 0,
        up_routed_format: 0,
        up_shared_format: 0,
        floor_slots: 128,
        chunk_slots: 128,
        tiles: MOE_TILES_ALL,
    };
    let d = moe_expert_gate_up(&gate_up, 8, tiles, 2).unwrap();
    assert_eq!(d.entry, "moe_expert_gguf_m8_gate_up_s0_softcap4_25");
    assert_eq!(d.slots, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]);
    assert_eq!(
        (d.addressed.as_slice(), d.bytes_slot, d.indirect),
        (&[13, 14][..], 12, Some((2, 8)))
    );
    assert_eq!(d.constants, uint_constants(&[0, 0, 0, 0]));
    let mixed = MoeGgufGateUpParams {
        up_routed_format: 3,
        up_shared_format: 3,
        ..gate_up
    };
    let d = moe_expert_gate_up(&mixed, 8, tiles, 1).unwrap();
    assert_eq!(d.entry, "moe_expert_gguf_m8_gate_up_mixed_s0_softcap4_25");
    assert_eq!(d.constants, uint_constants(&[0, 3, 3, 3]));
    let down = MoeGgufExpertParams {
        input_size: 1152,
        output_size: 2304,
        experts: 128,
        routed_format: 4,
        shared_format: 4,
        floor_slots: 128,
        chunk_slots: 128,
        tiles: MOE_TILES_ALL,
    };
    let d = moe_expert_down(&down, 8, tiles, 1).unwrap();
    assert_eq!(d.slots, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]);
    assert_eq!(
        (d.addressed.as_slice(), d.bytes_slot, d.indirect),
        (&[10][..], 9, Some((2, 20)))
    );
    assert_eq!((d.grid, d.threads), ([36, tiles, 1], [64, 1, 1]));
    assert!(moe_expert_down(&down, 16, tiles, 1).is_err());
}

#[test]
fn vendored_dense_dispatches() {
    let kv = [
        Segment {
            columns: 1024,
            format: Q80,
            offset: 0,
        },
        Segment {
            columns: 1024,
            format: Q80,
            offset: 1024,
        },
    ];
    let d = gguf_decode_sg_fused(
        1,
        &GgufDecodeFusedParams::new(2304, 8, 2048, &kv, 0).unwrap(),
    )
    .unwrap();
    assert_eq!(
        (d.bytes_slot, d.grid, d.threads),
        (14, [32, 8, 1], [128, 1, 1])
    );
    let q = [
        Segment {
            columns: 3072,
            format: Q80,
            offset: 0,
        },
        Segment {
            columns: 3072,
            format: Q80,
            offset: 3072,
        },
        Segment {
            columns: 384,
            format: Q80,
            offset: 6144,
        },
    ];
    let d = gguf_decode_fused(
        8,
        &GgufDecodeFusedParams::new(2304, 2, 6592, &q, 1).unwrap(),
    )
    .unwrap();
    assert_eq!((d.bytes_slot, d.grid), (13, [102, 2, 1]));
    assert_eq!(d.constants, uint_constants(&[Q80, Q80, Q80]));
    let out = GgufDecodeParams {
        input_size: 3072,
        splits: 4,
        out_stride: 2304,
        out_offset: 0,
        spread: 1,
    };
    let d = gguf_decode(Q80_TOKEN, 16, Destination::Bf16, &out, 2304).unwrap();
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid),
        ("gguf_decode_q80_m16_a", 8, [36, 4, 1])
    );
    let d = gguf_decode_sg(
        Q80_TOKEN,
        1,
        Destination::Bf16,
        &GgufDecodeParams { splits: 8, ..out },
        2304,
    )
    .unwrap();
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid),
        ("gguf_decode_sg_q80_l1_a", 9, [36, 8, 1])
    );
    let head = GgufDecodeParams {
        input_size: 2304,
        splits: 1,
        out_stride: 201024,
        out_offset: 0,
        spread: 1,
    };
    let d = gguf_decode(Q80_TOKEN, staged_rows(5), Destination::F32, &head, 201024).unwrap();
    assert_eq!(
        (d.entry.as_str(), d.grid),
        ("gguf_decode_q80_m8_a_f32", [3141, 1, 1])
    );
    assert!(
        gguf_decode(
            Q80_TOKEN,
            8,
            Destination::Bf16,
            &GgufDecodeParams { splits: 5, ..out },
            2304
        )
        .is_err()
    );
    let d = gguf_column(
        &GgufColumnParams {
            rows: 1,
            input_size: 2304,
            out_stride: 201024,
            out_offset: 0,
            format: Q80,
        },
        Destination::F32,
        201024,
    )
    .unwrap();
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid),
        ("gguf_column_a_f32", 5, [6282, 1, 1])
    );
    let d = gguf_column_fused(&GgufColumnFusedParams::new(2, 2304, 6592, &q).unwrap()).unwrap();
    assert_eq!((d.bytes_slot, d.grid), (11, [204, 1, 1]));
    let d = decode_linear_gguf_prepare(2304, 2).unwrap();
    assert_eq!(
        (d.bytes, d.bytes_slot, d.grid, d.threads),
        (width_bytes(2304), 3, [72, 2, 1], [128, 1, 1])
    );
    assert_eq!(
        table16_bytes(2304, 1),
        (2304 * 8 * 2, (144 * 8 + 72 * 8) * 4)
    );
}

#[test]
fn vendored_small_dispatches() {
    let d = gguf_embed(
        Q80_TOKEN,
        &GgufEmbedParams {
            rows: 5,
            vocabulary: 201024,
            hidden: 2304,
        },
    );
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid),
        ("gguf_embed_q80", 3, [45, 1, 1])
    );
    let d = norm_rms(5, 2304);
    assert_eq!(
        (d.bytes_slot, d.grid, d.bytes),
        (3, [5, 1, 1], width_bytes(2304))
    );
    let d = copy_rows_bf16(&RowCopyParams {
        rows: 5,
        width: 2304,
        ..Default::default()
    });
    assert_eq!((d.bytes_slot, d.grid), (2, [45, 1, 1]));
    let d = logit_finish(&LogitFinishParams {
        rows: 5,
        vocabulary: 201024,
        tokens: 200058,
        softcap: 30.0,
    });
    assert_eq!((d.bytes_slot, d.grid), (1, [786, 5, 1]));
    let sampling = TargetSamplingParams::greedy(201024, &[]).unwrap();
    let d = decode_sample_argmax_sharded(&sampling, 8, 1).unwrap();
    assert_eq!((d.bytes_slot, d.grid), (5, [128, 1, 1]));
    let d = decode_sample_argmax_reduce(&sampling, 8, 1).unwrap();
    assert_eq!(
        (d.bytes_slot, d.grid, d.threads),
        (4, [8, 1, 1], [32, 1, 1])
    );
    let d = decode_accept_tree(&AcceptBatchParams::default(), 1).unwrap();
    assert_eq!((d.bytes_slot, d.grid, d.threads), (7, [1, 1, 1], [1, 1, 1]));
    let d = moe_route_select(&MoeBiasedRouteParams {
        route: MoeRouteParams {
            rows: 5,
            input_size: 2304,
            experts: 128,
            top_k: 8,
        },
        scale: 8.0,
    })
    .unwrap();
    assert_eq!(
        (d.bytes_slot, d.grid, d.threads),
        (4, [5, 1, 1], [256, 1, 1])
    );
    let d = moe_group_routes_verified(&MoeGroupParams {
        rows: 5,
        top_k: 8,
        tile_rows: 8,
        experts: 128,
        gate_up_columns: 36,
        down_columns: 36,
        streamed: 0,
        lanes: 1,
    })
    .unwrap();
    assert_eq!((d.bytes_slot, d.grid), (8, [1, 1, 1]));
    let d = moe_gather_rows(
        &MoeGatherParams {
            input_size: 2304,
            routes_per_row: 8,
        },
        40,
    )
    .unwrap();
    assert_eq!((d.bytes_slot, d.grid), (5, [40, 9, 1]));
}

#[test]
fn vendored_draft_params() {
    // The sizes of the reference's static assertions (DraftAttention.h,
    // RoPE.h, Sampling.h).
    let b = sized(&DraftConvBatchParams {
        finish: 1,
        hidden: 2304,
    });
    assert_eq!((word(&b, 0), word(&b, 4)), (1, 2304));
    let b = sized(&draft_attention_params(777));
    assert_eq!(DraftAttentionBatchParams::SIZE, 40);
    assert_eq!((word(&b, 0), word(&b, 4), word(&b, 8)), (2048, 1, 777));
    assert_eq!(word(&b, 12), 0);
    assert_eq!(DraftContextBatchParams::SIZE, 32);
    assert_eq!(RopeRangeParams::SIZE, 8);
    let b = sized(&SelectorBatchParams::whole(42, 201024));
    assert_eq!(SelectorBatchParams::SIZE, 52);
    assert_eq!(word(&b, 0), 42);
    assert_eq!(
        (
            word(&b, 32),
            word(&b, 36),
            word(&b, 40),
            word(&b, 44),
            word(&b, 48)
        ),
        (1, 201024, 201024, 0, 0)
    );
}

#[test]
fn vendored_draft_geometry() {
    assert_eq!(draft_dynamic_width(2304), 576);
    assert_eq!(DRAFT_QKV_WIDTH, 6144);
    // Query rows, then a 32 x 130 fp32 partial per (head, split).
    assert_eq!(
        draft_grouped_queries_bytes(1),
        8 * 4096 * 2 + 8 * 4 * 32 * 130 * 4
    );
    assert_eq!(draft_ring_bytes(), 8 * 2048 * 128 * 2);
    // Partial IDs, values with the edge tables, candidates, unary.
    assert_eq!(
        draft_selector_elements(7),
        [7 * 16 * 8, 7 * 16 * 24, 112, 112]
    );
    let f = draft_inverse_frequencies(10000.0);
    assert_eq!(f.len(), 64);
    assert_eq!(f[0], 1.0);
    assert_eq!(f[32], 10000f32.powf(-0.5));
}

#[test]
fn vendored_draft_dispatches() -> Result<()> {
    let conv = draft_conv(DraftConvolutionStage::Residual, 2304)?;
    assert_eq!(conv.source, VendoredSource::Draft);
    assert_eq!((conv.grid, conv.bytes_slot), ([72, 1, 1], 5));
    assert_eq!(word(&conv.bytes, 0), 1);
    assert!(draft_conv(DraftConvolutionStage::Prepare, 2300).is_err());
    let qkv = draft_attention_qkv();
    assert_eq!(
        (qkv.grid, qkv.slots.len(), qkv.bytes.len()),
        ([320, 1, 1], 8, 0)
    );
    let p = draft_attention_params(100);
    let split = draft_attention_split(&p)?;
    assert_eq!((split.grid, split.bytes_slot), ([8, 1, 4], 19));
    let reduce = draft_attention_reduce(&p);
    assert_eq!((reduce.grid, reduce.bytes_slot), ([8, 1, 1], 1));
    let commit = draft_context_kv_commit(5, 16)?;
    assert_eq!((commit.grid, commit.bytes_slot), ([128, 1, 1], 22));
    assert_eq!(word(&commit.bytes, 0), 5);
    assert!(draft_context_kv_commit(5, 17).is_err());
    let rope = rope_draft_range(16, 9);
    assert_eq!((rope.source, rope.grid), (VendoredSource::Rope, [4, 1, 1]));
    let s = SelectorBatchParams::whole(1, 201024);
    let top = draft_select_top16(&s)?;
    assert_eq!((top.grid, top.bytes_slot), ([56, 1, 1], 3));
    let edges = draft_select_edges(&s)?;
    assert_eq!(
        (edges.grid, edges.threads, edges.bytes_slot),
        ([7, 1, 1], [256, 1, 1], 7)
    );
    let table = norm_rms_table16(8, 2304)?;
    assert_eq!(
        (table.entry.as_str(), table.bytes_slot),
        ("norm_rms_table16_decode", 5)
    );
    assert_eq!(norm_rms_bf16(8, 2304).entry, "norm_rms");
    Ok(())
}

#[test]
fn vendored_gate_up_and_epilogue_dispatches() -> Result<()> {
    let p = GgufDecodeParams {
        input_size: 2304,
        splits: 2,
        out_stride: 6912,
        out_offset: 0,
        spread: 1,
    };
    let pair = gguf_decode_gate_up("af4g64", 8, &p, 6912)?;
    assert_eq!(pair.entry, "gguf_decode_af4g64_m8_gate_up");
    assert_eq!(
        (pair.grid, pair.threads, pair.bytes_slot),
        ([216, 2, 1], [32, 1, 1], 10)
    );
    let up = gguf_decode_epilogue("af4g64", 16, Epilogue::UpWithGate, &p, 6912)?;
    assert_eq!(
        (up.entry.as_str(), up.grid),
        ("gguf_decode_af4g64_m16_g", [108, 2, 1])
    );
    let gate = gguf_decode_sg_epilogue("af4g64", 1, Epilogue::Plain(Destination::Bf16), &p, 6912)?;
    assert_eq!(gate.entry, "gguf_decode_sg_af4g64_l1_a");
    let column = GgufColumnParams {
        rows: 2,
        input_size: 2304,
        out_stride: 6912,
        out_offset: 0,
        format: 26,
    };
    let c = gguf_column_gate_up(&column, 6912)?;
    assert_eq!((c.grid, c.bytes_slot), ([216, 1, 1], 8));
    Ok(())
}

#[test]
fn vendored_prompt_params() {
    let segments = GgufPrefillSegmentsParams {
        input_size: 2304,
        rows: 300,
        out_stride: 6592,
        cols: [3072, 3072, 384],
        offset: [0, 3072, 6144],
    };
    let bytes = sized(&segments);
    assert_eq!((word(&bytes, 12), word(&bytes, 32)), (3072, 6144));
    assert_eq!(GgufPrefillSegmentsParams::SIZE, 36);
    let quantize = GgufPrefillInt8QuantizeParams {
        input_size: 2304,
        rows: 300,
        storage_rows: 384,
    };
    assert_eq!(word(&sized(&quantize), 8), 384);
    assert_eq!(GgufPrefillInt8QuantizeParams::SIZE, 12);
    let int8 = GgufPrefillInt8Params {
        input_size: 2304,
        rows: 300,
        storage_rows: 384,
        out_stride: 6592,
        out_offset: 3072,
        plane_input_size: 2304,
        column_tiles: 48,
    };
    assert_eq!(word(&sized(&int8), 24), 48);
    assert_eq!(GgufPrefillInt8Params::SIZE, 28);
    let context = DraftContextParams {
        tokens: 300,
        start_position: 17,
    };
    assert_eq!(word(&sized(&context), 4), 17);
    assert_eq!(DraftContextParams::SIZE, 8);
}

#[test]
fn vendored_prompt_dispatches() {
    // The relative attention of a 300-row chunk after 40 tokens.
    let mut p = AttentionParams {
        sequences: 1,
        rows: 300,
        tiles: 38,
        ..Default::default()
    };
    p.sequence[0] = prompt_sequence(40, 300, 64, 2);
    assert_eq!(
        (p.sequence[0].chunk_stride, p.sequence[0].row_bound),
        (320, 300)
    );
    let d = relative_attention_prepare(&p, true).unwrap();
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid, d.threads),
        (
            "relative_attention_q8_prepare_scaled",
            18,
            [32, 300, 1],
            [128, 1, 1]
        )
    );
    let d = relative_attention_split(&p, true, true, 2).unwrap();
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid),
        ("relative_attention_q8q_split_w1023", 15, [8, 38, 2])
    );
    let d = relative_attention_reduce(&p, false, true).unwrap();
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid, d.threads),
        (
            "relative_attention_reduce_table16",
            8,
            [8, 24, 38],
            [128, 1, 1]
        )
    );
    p.tiles = 19;
    let d = relative_attention_direct(&p, false, false).unwrap();
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid),
        ("relative_attention_q8_direct", 19, [8, 19, 1])
    );
    assert_eq!(sequence_workspace_bytes(), (256 * 192 * 512, 256 * 192 * 8));
    assert_eq!(
        staging_bytes(32),
        (8 * 32 * 3 * 256, 8 * 32 * 3 * 64, 8 * 32 * 3 * 4)
    );
    // One-tile sequences take 32 to 128 splits; longer ones share them.
    assert_eq!(sequence_splits(1, 1, 0, 32), 32);
    assert_eq!(sequence_splits(1, 4000, 0, 32), 128);
    assert_eq!(sequence_splits(1, 4000, 1023, 32), 34);
    assert_eq!(sequence_splits(256, 64, 0, 32), 1);
    assert_eq!(sequence_splits(6, 2, 0, 32), 5);

    // Mixes past 64 rows and gates past 128 rows run the row kernels.
    let mix = BlockResidualMixParams {
        rows: 300,
        hidden: 2304,
        stack: 2,
        queries: 3,
        stack_plane: 384 * 2304,
        scale: 1.0,
    };
    let d = block_residual_mix_norm(&mix, MixCurrent::F32).unwrap();
    assert_eq!(
        (d.source, d.entry.as_str(), d.grid, d.threads),
        (
            VendoredSource::BlockResidualMixRows,
            "block_residual_mix_norm_rows_f32",
            [300, 1, 1],
            [256, 1, 1]
        )
    );
    let gate = MessageGateFusedParams {
        rows: 300,
        hidden: 2304,
        x_stride: 3 * 2304,
        message_stride: 9216,
        rank: 64,
        gates: 3,
        x_columns: [0, 2304, 4608],
        message_columns: [0, 2304, 4608],
    };
    assert!(message_gate_apply(&gate, GateWeights::F32).is_err());
    let d = message_gate_rows(&gate, GateWeights::Bf16).unwrap();
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid),
        ("message_gate_rows_bf16", 20, [10, 1, 3])
    );

    // The chunk's experts and its combine.
    let group = MoeGroupParams {
        rows: 300,
        top_k: 8,
        tile_rows: 32,
        experts: 128,
        gate_up_columns: 36,
        down_columns: 36,
        streamed: 0,
        lanes: 0,
    };
    let d = moe_group_routes(&group).unwrap();
    assert_eq!((d.entry.as_str(), d.bytes_slot), ("moe_group_routes_s0", 7));
    assert!(moe_group_routes(&MoeGroupParams { lanes: 1, ..group }).is_err());
    let d = moe_combine(&MoeCombineParams {
        rows: 300,
        hidden_size: 2304,
        routes_per_row: 8,
    })
    .unwrap();
    assert_eq!((d.bytes_slot, d.grid), (5, [300, 9, 1]));

    // Float, prefill and int8 prefill tiles.
    let float = GgufFloatParams {
        rows: 300,
        input_size: 2304,
        output_size: [128, 0],
        out_stride: [256, 0],
        out_offset: [0, 0],
    };
    let d = gguf_float_na(true, true, &float).unwrap();
    assert_eq!(
        (d.entry.as_str(), d.grid, d.threads),
        ("gguf_float_na_f32_f32", [4, 5, 1], [128, 1, 1])
    );
    let segments = GgufPrefillSegmentsParams {
        input_size: 2304,
        rows: 300,
        out_stride: 6592,
        cols: [3072, 3072, 384],
        offset: [0, 3072, 6144],
    };
    let d = gguf_prefill("q80", &segments, 384).unwrap();
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid),
        ("gguf_prefill_q80_a", 11, [3, 102, 1])
    );
    let d = gguf_prefill_int8_quantize(&GgufPrefillInt8QuantizeParams {
        input_size: 2304,
        rows: 300,
        storage_rows: 384,
    })
    .unwrap();
    assert_eq!((d.grid, d.threads), ([5, 300, 1], [256, 1, 1]));
    let d = gguf_prefill_int8(
        "q80",
        &GgufPrefillInt8Params {
            input_size: 2304,
            rows: 300,
            storage_rows: 384,
            out_stride: 1024,
            out_offset: 0,
            plane_input_size: 2304,
            column_tiles: 16,
        },
    )
    .unwrap();
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid, d.threads),
        ("gguf_prefill_int8_q80_a", 7, [16, 10, 1], [64, 1, 1])
    );
    assert_eq!(prefill_int8_bytes(3072, 128), (3072 * 128, 48 * 128 * 8));
    let d = prefill_draft_context_kv(300, 17);
    assert_eq!(
        (d.entry.as_str(), d.bytes_slot, d.grid),
        ("prefill_draft_context_kv", 6, [2400, 1, 1])
    );
}
