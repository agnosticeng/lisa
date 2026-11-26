//! The device tree against a host transcription of the vendored reference's
//! greedy tree growth and verify-row layout.
use super::*;

const P: usize = TREE_POSITIONS;
const K: usize = TREE_CANDIDATES;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A lattice in sorted candidate order: IDs `[P][K]`, logits and edges.
struct Lattice {
    ids: Vec<u32>,
    logits: Vec<f32>,
    edges: Vec<f32>,
}

fn lattice(seed: u64) -> Lattice {
    let mut rng = Rng(seed);
    let mut ids = vec![];
    let mut logits = vec![];
    for p in 0..P {
        // Distinct tokens per position; some shared across positions so that
        // a chain can agree with the draft.
        for i in 0..K {
            ids.push(1000 + (p as u32 % 3) * 100 + i as u32);
            logits.push(20.0 - 1.3 * i as f32 + 0.4 * rng.unit());
        }
    }
    let edges = (0..P * K * K).map(|_| 6.0 * rng.unit() - 3.0).collect();
    Lattice { ids, logits, edges }
}

struct Model {
    slope: f32,
    intercepts: [f32; P],
    softcap: f32,
}

fn estimate(m: &Model, q: f32, p: usize) -> f32 {
    let q = q.clamp(1e-6, 1.0 - 1e-6);
    let z = m.slope * (q / (1.0 - q)).ln() + m.intercepts[p];
    1.0 / (1.0 + (-z).exp())
}

/// Host transcription of `draft_select_tree` (greedy) and
/// `draft_verify_rows` for one lane: verify tokens and parents.
fn reference(
    l: &Lattice,
    m: &Model,
    chain: &[u32],
    rates: &[f32],
    params: &TreeParams,
) -> (Vec<u32>, Vec<i32>) {
    // Probabilities and likeliest children per (position, predecessor).
    let mut prob = vec![[0f32; K]; P * K];
    let mut likeliest = vec![[0usize; TREE_CHILDREN]; P * K];
    for p in 0..P {
        for a in 0..if p > 0 { K } else { 1 } {
            let score: Vec<f32> = (0..K)
                .map(|i| {
                    let s = l.logits[p * K + i] + l.edges[(p * K + a) * K + i];
                    m.softcap * (s / m.softcap).tanh()
                })
                .collect();
            let best = score.iter().cloned().fold(f32::MIN, f32::max);
            let w: Vec<f32> = score.iter().map(|s| (s - best).exp()).collect();
            let mass: f32 = w.iter().sum();
            for i in 0..K {
                prob[p * K + a][i] = w[i] / mass;
            }
            let mut order: Vec<usize> = (0..K).collect();
            order.sort_by(|&x, &y| score[y].total_cmp(&score[x]).then(x.cmp(&y)));
            likeliest[p * K + a].copy_from_slice(&order[..TREE_CHILDREN]);
        }
    }
    let agreement = |token: u32, d: usize| -> usize {
        if d >= P {
            3
        } else if token == l.ids[d * K] {
            0
        } else if (1..K).any(|i| l.ids[d * K + i] == token) {
            1
        } else {
            2
        }
    };
    let rate = |d: usize| rates[d * TREE_AGREEMENTS + agreement(chain[d], d)];
    // (score, order, chain, parent, position, index)
    let mut heap: Vec<(f32, usize, bool, usize, usize, usize)> = vec![];
    let mut pushed = 0;
    let mut push = |heap: &mut Vec<_>, s: f32, c: bool, parent, pos, idx| {
        heap.push((s, pushed, c, parent, pos, idx));
        pushed += 1;
    };
    struct Row {
        token: u32,
        parent: usize,
        on_chain: bool,
        q: f32,
        estimate: f32,
        next: [f32; K],
    }
    let mut rows = vec![Row {
        token: params.anchor,
        parent: usize::MAX,
        on_chain: true,
        q: 1.0,
        estimate: 1.0,
        next: [0.; K],
    }];
    let expand = |rows: &mut Vec<Row>,
                  heap: &mut Vec<_>,
                  push: &mut dyn FnMut(&mut Vec<_>, f32, bool, usize, usize, usize),
                  row: usize,
                  pos: usize,
                  pred: usize,
                  score: f32,
                  chain_token: u32,
                  chain_rate: f32| {
        let q = prob[pos * K + pred];
        let mass: f32 = q.iter().sum();
        for i in 0..K {
            rows[row].next[i] = q[i] / mass;
        }
        let (mut bound, mut left, mut reserved) = (score, 1.0 - chain_rate, chain_rate);
        for child in 0..params.children as usize {
            let drawn = likeliest[pos * K + pred][child];
            if q[drawn] <= 0.0 {
                return;
            }
            let joins = l.ids[pos * K + drawn] == chain_token;
            let cap = if joins { left + reserved } else { left };
            let own = estimate(m, rows[row].next[drawn], pos).min(cap);
            left = if joins {
                cap - own.max(reserved)
            } else {
                left - own
            };
            if joins {
                reserved = 0.0;
            }
            bound = bound.min(score * own);
            push(heap, bound, false, row, pos, drawn);
        }
    };
    let chained = params.chain as usize;
    if params.depth > 0 {
        let r = if chained > 0 { rate(0) } else { 0.0 };
        let token = if chained > 0 { chain[0] } else { u32::MAX };
        expand(&mut rows, &mut heap, &mut push, 0, 0, 0, 1.0, token, r);
        if chained > 0 {
            push(&mut heap, r, true, 0, 0, 0);
        }
    }
    let most = TREE_ROWS.min(params.depth as usize + 1);
    while !heap.is_empty() && rows.len() < most {
        let best = (0..heap.len())
            .min_by(|&a, &b| {
                heap[b]
                    .0
                    .total_cmp(&heap[a].0)
                    .then(heap[a].1.cmp(&heap[b].1))
            })
            .unwrap();
        let (score, _, from_chain, parent, pos, idx) = heap.swap_remove(best);
        let token = if from_chain {
            chain[pos]
        } else {
            l.ids[pos * K + idx]
        };
        let row =
            match (1..rows.len()).find(|&o| rows[o].parent == parent && rows[o].token == token) {
                Some(o) => {
                    if !from_chain && rows[o].q < 0.0 {
                        rows[o].q = rows[parent].next[idx];
                    }
                    o
                }
                None => {
                    let on_chain = rows[parent].on_chain && pos < chained && token == chain[pos];
                    let q = if from_chain {
                        -1.0
                    } else {
                        rows[parent].next[idx]
                    };
                    rows.push(Row {
                        token,
                        parent,
                        on_chain,
                        q,
                        estimate: score,
                        next: [0.; K],
                    });
                    rows.len() - 1
                }
            };
        if pos + 1 >= params.depth as usize {
            continue;
        }
        let chain_child = rows[row].on_chain && pos + 1 < chained;
        if !from_chain && pos + 1 < P {
            let (t, r) = if chain_child {
                (chain[pos + 1], rate(pos + 1))
            } else {
                (u32::MAX, 0.0)
            };
            expand(
                &mut rows,
                &mut heap,
                &mut push,
                row,
                pos + 1,
                idx,
                score,
                t,
                r,
            );
        } else if from_chain && pos + 1 < chained {
            push(&mut heap, score * rate(pos + 1), true, row, pos + 1, 0);
        }
    }
    // Verify rows: the best estimates, then preorder.
    let mut ranked: Vec<usize> = (1..rows.len()).collect();
    ranked.sort_by(|&a, &b| rows[b].estimate.total_cmp(&rows[a].estimate));
    let mut take = 1;
    for &r in &ranked {
        if take >= params.rows as usize || rows[r].estimate < params.floor {
            break;
        }
        if take == params.lane_cap as usize {
            continue;
        }
        take += 1;
    }
    let mut order = vec![];
    let mut stack = vec![0usize];
    while let Some(r) = stack.pop() {
        order.push(r);
        for c in (r + 1..take).rev() {
            if rows[c].parent == r {
                stack.push(c);
            }
        }
    }
    let placed = |r: usize| order.iter().position(|&o| o == r).unwrap() as i32;
    let mut tokens: Vec<u32> = order.iter().map(|&r| rows[r].token).collect();
    let mut parents: Vec<i32> = order
        .iter()
        .map(|&r| if r == 0 { -1 } else { placed(rows[r].parent) })
        .collect();
    while tokens.len() < params.rows as usize {
        tokens.push(params.anchor);
        parents.push(-1);
    }
    (tokens, parents)
}

#[test]
fn device_tree_matches_the_reference_growth() {
    let stream = Stream::gpu();
    let model = Model {
        slope: 0.7,
        intercepts: [0.0, -0.35, -0.8, -0.75, -0.75, -0.85, -0.45],
        softcap: 30.0,
    };
    for (seed, chain_len, rows, floor) in [
        (1u64, 0usize, 16u32, 0.0f32),
        (2, 4, 16, 0.0),
        (3, 8, 12, 0.0),
        (4, 3, 8, 0.05),
        (5, 15, 16, 0.0),
    ] {
        let l = lattice(seed);
        // The chain follows the draft's likeliest candidate at its first
        // positions and leaves it after.
        let chain: Vec<u32> = (0..TREE_CHAIN_TOKENS)
            .map(|d| if d < 2 { l.ids[d * K] } else { 5000 + d as u32 })
            .collect();
        let rates: Vec<f32> = (0..TREE_CHAIN_TOKENS * TREE_AGREEMENTS)
            .map(|i| [0.9, 0.6, 0.4, 0.3][i % 4] * 0.97f32.powi((i / 4) as i32))
            .collect();
        let params = TreeParams {
            anchor: 77,
            depth: 15,
            chain: chain_len as u32,
            children: 4,
            softcap: model.softcap,
            slope: model.slope,
            intercepts: model.intercepts,
            rows,
            lane_cap: 16,
            floor,
            selected: false,
        };
        // Shuffle each position's candidates; the glue restores the order.
        let mut rng = Rng(seed * 7 + 1);
        let mut perm = vec![];
        for _ in 0..P {
            let mut p: Vec<usize> = (0..K).collect();
            for i in (1..K).rev() {
                p.swap(i, (rng.next() % (i as u64 + 1)) as usize);
            }
            perm.push(p);
        }
        let mut ids = vec![0i32; P * K];
        let mut logits = vec![0f32; P * K];
        let mut edges = vec![0f32; P * K * K];
        for p in 0..P {
            for (shuffled, &sorted) in perm[p].iter().enumerate() {
                ids[p * K + shuffled] = l.ids[p * K + sorted] as i32;
                logits[p * K + shuffled] = l.logits[p * K + sorted];
            }
            for a in 0..K {
                let sa = if p > 0 { perm[p - 1][a] } else { 0 };
                for (shuffled, &sorted) in perm[p].iter().enumerate() {
                    edges[(p * K + a) * K + shuffled] = l.edges[(p * K + sa) * K + sorted];
                }
            }
        }
        let tree = draft_tree(
            &Array::from_slice(&ids, &[P as i32, K as i32]),
            &Array::from_slice(&logits, &[P as i32, K as i32]),
            &Array::from_slice(&edges, &[P as i32, K as i32, K as i32]),
            &Array::from_slice(
                &chain.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                &[TREE_CHAIN_TOKENS as i32],
            ),
            &Array::from_slice(&rates, &[TREE_CHAIN_TOKENS as i32, TREE_AGREEMENTS as i32]),
            &params,
            &stream,
        )
        .unwrap();
        let (tokens, parents) = reference(&l, &model, &chain, &rates, &params);
        let got_tokens: Vec<u32> = tree
            .ids
            .as_slice::<i32>()
            .iter()
            .map(|&t| t as u32)
            .collect();
        let got = tree.tree.as_slice::<i32>().to_vec();
        assert_eq!(got_tokens, tokens, "seed {seed}");
        assert_eq!(got[..rows as usize], parents[..], "seed {seed}");
        // Depths follow the parents.
        for r in 0..rows as usize {
            let d = if parents[r] < 0 {
                0
            } else {
                got[rows as usize + parents[r] as usize] + 1
            };
            assert_eq!(got[rows as usize + r], d, "seed {seed} row {r}");
        }
    }
}
