"""Apply identical scoped edits to the two existing benchmark entry points."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for name in ['run_video.py','run_video_power.py']:
    path=ROOT/'src/v2/deploy'/name; s=path.read_text(encoding='utf8')
    def replace(old,new):
        global s
        assert s.count(old)==1,(name,old,s.count(old))
        s=s.replace(old,new)
    replace('from src.v2.dataset import crop as C','from src.v2.dataset import crop as C\nfrom src.v2.deploy import inference_policy as IP')
    replace('    ap.add_argument("--onnx-dir", default=ONNX_DIR)',
            '    ap.add_argument("--onnx-dir", default=ONNX_DIR)\n    ap.add_argument("--model-contract", default=None, help="Graph-hash-bound validation threshold JSON; auto-detect contract.json if omitted")')
    replace('    args = ap.parse_args()', '    args = ap.parse_args()\n    if args.head_stride < 1:\n        ap.error("--head-stride must be positive")')
    line='    t_read, t_detect, t_crop, t_encode, t_head, t_e2e = ([] for _ in range(6))'
    replace(line,'''    model_dir = args.onnx_dir if args.mode == "ours" else os.path.join(args.onnx_dir, args.mode)
    contract = None
    if head_sess is not None:
        graph_names = ["encoder.onnx" if args.mode == "ours" else "backbone.onnx", "head.onnx"]
        contract = IP.load_contract(model_dir, [os.path.join(model_dir, n) for n in graph_names], args.model_contract)
    random_weights = contract is None and args.mode != "ear"
    n_positive_decisions = 0
'''+line)
    replace('if len(ring) == EVENT_LEN and n_frames % args.head_stride == 0:',
            'if IP.ring_ready(ring) and n_frames % args.head_stride == 0:')
    replace('                probs.append(p)', '                probs.append(p)\n                if contract is not None:\n                    n_positive_decisions += int(p >= contract["threshold_probability"])')
    replace('if len(ear_ring) == EVENT_LEN and n_frames % args.head_stride == 0:',
            'if (len(ear_ring) == EVENT_LEN and IP.eligible_mask(np.isfinite(ear_ring))\n                    and n_frames % args.head_stride == 0):')
    start=s.index('    if random_weights:\n        warn.append('); end=s.index('\n',s.index('"이 런의 판정 출력(probs)은 의미가 없다")',start))
    s=s[:start]+'''    if random_weights:
        warn.append("검증된 모델 계약이 없어 가중치 출처·validation 임계값은 미확인. 정확도 주장을 하지 않는다")'''+s[end:]
    replace('        "weights": "random (latency only)" if random_weights else "trained",',
            '''        "weights": "rule (no weights)" if args.mode == "ear" else "unverified (latency only)" if random_weights else "trained (graph hash verified)",
        "inference_policy": IP.metadata(contract),
        "n_positive_decisions": n_positive_decisions if contract is not None else None,''')
    path.write_text(s,encoding='utf8')
