import fs from 'node:fs/promises';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const OUT = 'C:/Users/sch/PycharmProjects/IEEE_sensors/docs/랩미팅_2026-08-05_v2.pptx';
const TMP = 'C:/Users/sch/PycharmProjects/IEEE_sensors/temp_ppt_build/output';
const W = 1280, H = 720;
const C = {
  navy: '#102B4E', blue: '#2E5EAA', sky: '#EAF1FB', pale: '#F6F8FC',
  ink: '#162235', muted: '#56657A', line: '#D7E0EE', teal: '#2E7D78',
  red: '#B94B4B', white: '#FFFFFF', gold: '#C18A20'
};

async function saveBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}
function shape(slide, geometry, x, y, w, h, fill, line = fill, radius = 'rounded-xl') {
  const config = {
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: 'solid', fill: line, width: line === 'none' ? 0 : 1 },
  };
  if (radius !== 'none') config.borderRadius = radius;
  return slide.shapes.add(config);
}
function text(slide, x, y, w, h, value, size = 24, color = C.ink, bold = false, align = 'left') {
  const t = slide.shapes.add({
    geometry: 'textbox', position: { left: x, top: y, width: w, height: h },
    fill: 'none', line: { style: 'solid', fill: 'none', width: 0 },
  });
  t.text = value;
  t.text.style = { fontSize: size, fontFace: 'Malgun Gothic', color, bold, alignment: align, margin: 0 };
  return t;
}
function rule(slide, x, y, w, h = 3, color = C.blue) { shape(slide, 'rect', x, y, w, h, color, color, 'none'); }
function footer(slide, n, source) {
  rule(slide, 72, 676, 1136, 1, C.line);
  text(slide, 72, 685, 900, 18, source, 12, C.muted);
  text(slide, 1164, 684, 44, 20, String(n).padStart(2, '0'), 13, C.navy, true, 'right');
}
function notes(slide, source) {
  slide.speakerNotes.textFrame.setText(`[Sources]\n${source}`);
  slide.speakerNotes.setVisible(true);
}
function title(slide, n, heading, kicker) {
  text(slide, 72, 44, 840, 24, kicker, 14, C.blue, true);
  text(slide, 72, 78, 1060, 52, heading, 38, C.navy, true);
  rule(slide, 72, 144, 72, 4, C.blue);
  text(slide, 1160, 48, 48, 22, String(n).padStart(2, '0'), 14, C.blue, true, 'right');
}
function metric(slide, x, y, w, label, value, sub, accent = C.blue) {
  shape(slide, 'roundRect', x, y, w, 138, C.white, C.line);
  rule(slide, x, y, 7, 138, accent);
  text(slide, x + 24, y + 22, w - 42, 22, label, 16, C.muted, true);
  text(slide, x + 24, y + 52, w - 42, 42, value, 34, C.navy, true);
  text(slide, x + 24, y + 102, w - 42, 20, sub, 14, C.muted);
}
function arrow(slide, x, y, w) {
  text(slide, x, y, w, 28, '→', 28, C.blue, true, 'center');
}

async function main() {
  await fs.mkdir(TMP, { recursive: true });
  const p = Presentation.create({ slideSize: { width: W, height: H } });

  // 1. Cover
  {
    const s = p.slides.add(); s.background.fill = C.white;
    shape(s, 'rect', 0, 0, 34, H, C.navy, C.navy, 'none');
    text(s, 82, 104, 700, 24, 'LAB MEETING  |  2026.08.05', 17, C.blue, true);
    text(s, 82, 158, 740, 154, 'v2 눈 깜빡임 인코더:\n유용성은 확보했고,\n신원 누출은 남아 있다', 44, C.navy, true);
    text(s, 86, 326, 760, 62, 'mEBAL2 57명에서 직접 지도학습한 16차원 벡터와\nEAR 베이스라인을 같은 조건에서 비교', 24, C.muted);
    rule(s, 86, 432, 270, 6, C.blue);
    text(s, 86, 470, 620, 26, '핵심 질문  |  깜빡임은 잘 잡는가?  신원은 얼마나 남는가?', 20, C.ink, true);
    shape(s, 'roundRect', 892, 114, 272, 330, C.sky, C.sky);
    text(s, 925, 160, 200, 30, 'v2 기준선', 20, C.blue, true, 'center');
    text(s, 922, 220, 212, 72, '57명\n27,758 events', 31, C.navy, true, 'center');
    text(s, 922, 326, 212, 42, 'vpres CNN · D=16', 17, C.muted, true, 'center');
    text(s, 82, 652, 660, 20, 'IEEE Sensors Letters 방향 점검', 14, C.muted);
    notes(s, 'docs/v2/PROTOCOL.md §0; docs/PROJECT.md §2; README.md');
  }

  // 2. Direction change
  {
    const s = p.slides.add(); s.background.fill = C.white;
    title(s, 2, 'v2는 인코더보다 검증 규칙을 먼저 고정했습니다', 'WHY V2 CHANGED');
    text(s, 88, 190, 438, 30, '이전 v1', 23, C.muted, true);
    text(s, 672, 190, 438, 30, '현재 v2', 23, C.blue, true);
    shape(s, 'roundRect', 72, 236, 460, 292, C.pale, C.pale);
    text(s, 104, 274, 378, 40, '8명 공개셋 · 128D 오토인코더', 25, C.navy, true);
    text(s, 104, 338, 374, 110, '• 데이터·분할·임계값이 현재 문제와 다름\n• EAR와의 비교 기준이 흔들림\n• 성능·프라이버시 결론을 이어 쓸 수 없음', 20, C.muted);
    shape(s, 'roundRect', 628, 236, 580, 292, C.sky, C.sky);
    text(s, 662, 274, 490, 40, '57명 mEBAL2 · 직접 지도학습 vpres D=16', 25, C.navy, true);
    text(s, 662, 338, 490, 122, '• 피험자 분리 5-fold × 3 seed\n• EAR는 유용성 베이스라인으로 고정\n• 신원 누출은 “숨겼다”가 아니라 “얼마나 남는가”로 측정', 20, C.ink);
    text(s, 552, 339, 62, 62, '→', 42, C.blue, true, 'center');
    text(s, 82, 572, 1080, 38, '이번 라운드의 결론: 표현 학습의 효과와 입력 자체의 누출을 분리해 보여주는 기준선을 만든다.', 23, C.navy, true);
    footer(s, 2, 'Source: docs/PROJECT.md §2; README.md; docs/v2/PROMPT_v2.md (consumed-history context)');
    notes(s, 'docs/PROJECT.md §2 and v1/v2 table; README.md “핵심 전환”; docs/v2/PROMPT_v2.md “소비 완료”.');
  }

  // 3. Data reality
  {
    const s = p.slides.add(); s.background.fill = C.white;
    title(s, 3, '먼저 데이터의 실제 구조와 오염을 바로잡았습니다', 'DATA AUDIT BEFORE MODELING');
    metric(s, 72, 194, 260, '원 배포본', '58명', 'mEBAL1 38 + 신규 mEBAL2 20', C.blue);
    metric(s, 356, 194, 260, '최종 분석 코호트', '57명', 'U18 제외', C.teal);
    metric(s, 640, 194, 260, '유효 이벤트', '27,758', '28,728 중 결측·좌석·U18 제외', C.gold);
    metric(s, 924, 194, 284, '입력 규모', '532,109', '64×160 눈 크롭 프레임', C.blue);
    rule(s, 72, 382, 1136, 1, C.line);
    text(s, 88, 420, 440, 28, '발견', 20, C.red, true);
    text(s, 88, 460, 468, 86, '2022 배치에는 여러 사람이 함께 촬영된 장면이 있어\n주 피험자가 아닌 얼굴이 선택되는 사례가 존재', 20, C.ink);
    text(s, 656, 420, 440, 28, '대응', 20, C.teal, true);
    text(s, 656, 460, 468, 86, '프레임별 화면 x 위치를 다시 추적해 좌석 이탈을 플래그하고\nU18 및 오염 이벤트를 분석에서 제외', 20, C.ink);
    text(s, 88, 590, 1060, 34, '중요: 모델 성능보다 먼저 “누구의 눈을 보고 있는가”를 검증한 뒤 평가 코호트를 고정했습니다.', 22, C.navy, true);
    footer(s, 3, 'Source: docs/v2/PROTOCOL.md §0, §3-bis, §3-ter; results/v2/apply_flags_report.json');
    notes(s, 'docs/v2/PROTOCOL.md §0 data table; §3-bis and §3-ter; results/v2/apply_flags_report.json.');
  }

  // 4. Pipeline
  {
    const s = p.slides.add(); s.background.fill = C.white;
    title(s, 4, '같은 입력에서 유용성과 신원 누출을 함께 측정합니다', 'CURRENT V2 PIPELINE');
    const y = 240, h = 150, xs = [72, 275, 478, 681, 884];
    const labels = [
      ['mEBAL2 이벤트', '19프레임\nblink / unblink'],
      ['품질·오염 필터', '결측 ≤5\n좌석 이탈·U18 제외'],
      ['눈 크롭·정규화', '64×160 · margin 2.2\nframe standardize'],
      ['vpres CNN', '프레임당\n16차원 벡터'],
      ['시간 판정 헤드', 'TCN + MLP\nblink 확률']
    ];
    labels.forEach((a, i) => {
      shape(s, 'roundRect', xs[i], y, 164, h, i === 3 ? C.sky : C.pale, i === 3 ? C.blue : C.line);
      text(s, xs[i] + 14, y + 28, 136, 31, a[0], 18, C.navy, true, 'center');
      text(s, xs[i] + 14, y + 74, 136, 54, a[1], 16, C.muted, false, 'center');
      if (i < 4) arrow(s, xs[i] + 166, y + 59, 35);
    });
    text(s, 72, 452, 1136, 26, '평가 설계', 20, C.blue, true);
    shape(s, 'roundRect', 72, 494, 532, 96, C.white, C.line);
    text(s, 96, 516, 486, 22, '유용성', 18, C.navy, true);
    text(s, 96, 548, 486, 24, '피험자 분리 5-fold × 3 seed · PR-AUC · EAR-head와 비교', 17, C.muted);
    shape(s, 'roundRect', 676, 494, 532, 96, C.white, C.line);
    text(s, 700, 516, 486, 22, '신원 누출', 18, C.navy, true);
    text(s, 700, 548, 486, 24, '시간 블록 MLP 재식별 · 균형 표집 · 랜덤 인코더와 비교', 17, C.muted);
    footer(s, 4, 'Source: src/v2/train_encoder.py; src/v2/model/encoder.py; docs/v2/PROTOCOL.md §§4, 7–9.');
    notes(s, 'src/v2/train_encoder.py; src/v2/model/encoder.py; docs/v2/PROTOCOL.md §§4, 7–9.');
  }

  // 5. Utility
  {
    const s = p.slides.add(); s.background.fill = C.white;
    title(s, 5, 'EAR-head 대비 유용성은 확보했습니다', 'UTILITY: SUBJECT-DISJOINT EVALUATION');
    text(s, 72, 190, 610, 36, '5-fold × 3 seed = 15개 독립 실행', 23, C.muted);
    const chartX=96, chartY=278, maxW=560;
    const bars=[['ours  |  vpres D=16',0.9886,C.blue],['EAR-head  |  동일 시간 판정 헤드',0.9724,C.teal],['EAR rule  |  규칙 기반',0.8931,C.gold]];
    bars.forEach((b,i)=>{
      const yy=chartY+i*82;
      text(s, chartX, yy, 360, 24, b[0], 18, C.ink, true);
      shape(s,'roundRect',chartX,yy+34,maxW,24,C.pale,C.pale);
      shape(s,'roundRect',chartX,yy+34,maxW*b[1],24,b[2],b[2]);
      text(s, chartX+maxW+18, yy+29, 120, 30, b[1].toFixed(4), 20, C.navy, true);
    });
    shape(s, 'roundRect', 808, 232, 400, 252, C.sky, C.sky);
    text(s, 842, 270, 330, 26, '정식 비교: ours − EAR-head', 18, C.muted, true, 'center');
    text(s, 838, 318, 340, 62, '+0.0151', 48, C.navy, true, 'center');
    text(s, 838, 392, 340, 30, '95% CI  [+0.0106, +0.0203]', 18, C.blue, true, 'center');
    text(s, 838, 438, 340, 24, 'δ = 0.02 기준: superior', 17, C.teal, true, 'center');
    text(s, 72, 582, 1100, 34, '해석: 같은 시간 판정 헤드와 비교해 얻은 차이입니다. 다만 안경 하위군은 표본이 작아 별도 결론을 내리지 않습니다.', 20, C.navy, true);
    footer(s, 5, 'Source: results/v2/train_encoder.json; results/v2/posthoc_subgroups.json; docs/v2/PROTOCOL.md §0, §9.');
    notes(s, 'results/v2/train_encoder.json; results/v2/posthoc_subgroups.json; docs/v2/PROTOCOL.md §0 and §9.');
  }

  // 6. Reidentification
  {
    const s = p.slides.add(); s.background.fill = C.white;
    title(s, 6, '학습된 16차원 벡터는 신원을 억제하지 못했습니다', 'PRIVACY AUDIT: RE-IDENTIFICATION');
    text(s, 72, 190, 660, 34, '57-way 시간 블록 MLP · 균형 표집 · chance = 0.0175', 21, C.muted);
    const items=[['원본 픽셀',0.9080,C.navy],['PCA-32',0.8669,C.blue],['ours  |  vpres D=16',0.6345,C.red],['랜덤 인코더 D=16',0.6261,C.gold],['광학 3차원',0.0825,C.teal],['EAR 스칼라',0.0548,C.muted]];
    items.forEach((a,i)=>{
      const yy=252+i*48;
      text(s, 88, yy, 280, 23, a[0], 17, C.ink, a[0].startsWith('ours'));
      shape(s,'roundRect',370,yy+2,500,18,C.pale,C.pale);
      shape(s,'roundRect',370,yy+2,500*a[1],18,a[2],a[2]);
      text(s,882,yy-2,106,25,a[1].toFixed(4),18,C.navy,true,'right');
    });
    shape(s, 'roundRect', 1010, 232, 198, 286, C.sky, C.sky);
    text(s, 1030, 272, 158, 28, '핵심 해석', 19, C.blue, true, 'center');
    text(s, 1030, 326, 158, 86, '학습 전 랜덤\n인코더와 거의\n같은 누출', 23, C.navy, true, 'center');
    text(s, 1030, 438, 158, 42, '“프라이버시 개선”\n주장 불가', 16, C.red, true, 'center');
    text(s, 72, 576, 1110, 36, '집계 공격: 관측 1개 0.9139 → 10개 0.9960 → 100개 1.0000.  차원 축소는 억제 기전이 아닙니다.', 20, C.navy, true);
    footer(s, 6, 'Source: results/v2/phase5_reid.json; docs/v2/PROTOCOL.md §0, Phase 5.');
    notes(s, 'results/v2/phase5_reid.json; docs/v2/PROTOCOL.md §0 Phase 5.');
  }

  // 7. Guardrails
  {
    const s = p.slides.add(); s.background.fill = C.white;
    title(s, 7, '평가 규칙을 먼저 고정하고 9개 게이트를 통과했습니다', 'REPRODUCIBILITY AND GUARDRAILS');
    const g=[['격자', '저 false-alarm 영역을\n40,001개 후보로 탐색'],['분할', '이벤트 수 + 수집 배치\n이중 층화·피험자 분리'],['선택', 'validation에서 선택하고\ntest에서만 보고'],['결정성', '실제 인코더까지\n같은 시드 bit-identical']];
    g.forEach((a,i)=>{
      const x=72+i*284;
      shape(s,'roundRect',x,224,244,210,i===3?C.sky:C.pale,i===3?C.blue:C.line);
      text(s,x+20,254,204,28,a[0],22,C.navy,true,'center');
      text(s,x+20,304,204,64,a[1],17,C.muted,false,'center');
      text(s,x+20,398,204,25,'PASS',18,C.teal,true,'center');
    });
    shape(s,'roundRect',72,506,1136,82,C.white,C.line);
    text(s,98,530,1090,26,'Phase −1 gate 9/9 PASS  |  GPU·seed·데이터를 포함한 실제 인코더 결정성까지 확인',21,C.navy,true,'center');
    text(s,98,628,1085,24,'단, 학습 원점수 보존과 깨끗한 커밋 상태는 다음 전체 재실행에서 더 강하게 봉인해야 합니다.',17,C.muted,false,'center');
    footer(s, 7, 'Source: results/v2/gate_minus1.json; docs/v2/PROTOCOL.md §§1–5, §11.');
    notes(s, 'results/v2/gate_minus1.json (latest PASS); docs/v2/PROTOCOL.md §§1–5 and §11.');
  }

  // 8. Honest boundary
  {
    const s = p.slides.add(); s.background.fill = C.white;
    title(s, 8, '현재 기준선이 말하는 것과 말하지 못하는 것을 분리합니다', 'HONEST BOUNDARY');
    shape(s,'roundRect',72,212,520,316,C.sky,C.sky);
    text(s,106,250,450,28,'이번 라운드에서 말할 수 있는 것',22,C.blue,true);
    text(s,108,306,438,150,'• EAR-head와 같은 판정 조건에서 utility 확보\n• 16차원 표현의 신원 누출을 정량화\n• 학습이 누출을 줄이지 못함을 기준선과 비교',20,C.ink);
    shape(s,'roundRect',688,212,520,316,C.pale,C.pale);
    text(s,722,250,450,28,'아직 말하면 안 되는 것',22,C.red,true);
    text(s,724,306,438,150,'• 프라이버시가 확보되었다\n• 안경군에서 우월하다\n• v2가 Pi5 30 fps를 만족한다\n• 표현 학습의 기여가 완전히 분리되었다',20,C.ink);
    text(s,72,578,1136,38,'따라서 다음 논문의 메시지는 “프라이버시 해결”이 아니라 “유용성·누출을 같은 조건에서 측정한 기준선”입니다.',22,C.navy,true,'center');
    footer(s, 8, 'Source: docs/v2/PROTOCOL.md §0, §§9–13; results/v2/posthoc_subgroups.json; docs/PI5_BENCHMARK.md.');
    notes(s, 'docs/v2/PROTOCOL.md §0, §§9–13; results/v2/posthoc_subgroups.json; docs/PI5_BENCHMARK.md.');
  }

  // 9. Next
  {
    const s = p.slides.add(); s.background.fill = C.white;
    title(s, 9, '다음은 “배포 가능성”과 “억제 필요성”을 분리해 닫는 일입니다', 'NEXT DECISIONS');
    const steps=[
      ['01', 'Phase 6 · Pi5 재측정', 'vpres D=16, 1280×720에서\np99 e2e ≤ 33.3 ms 확인'],
      ['02', '대조군 2·3·4', '표현 자체와 학습 목적함수의\n기여를 분리'],
      ['03', '신원 억제 방향 결정', '현재 0.6345 → 목표 < 0.086\n박사님 확인 후 후속 연구로']
    ];
    steps.forEach((a,i)=>{
      const x=72+i*380;
      text(s,x,216,56,38,a[0],28,C.blue,true);
      shape(s,'roundRect',x,276,330,190,i===0?C.sky:C.pale,i===0?C.blue:C.line);
      text(s,x+24,306,282,36,a[1],22,C.navy,true);
      text(s,x+24,366,278,60,a[2],18,C.muted);
    });
    rule(s,72,540,1136,1,C.line);
    text(s,72,572,1136,48,'랩미팅에서 필요한 판단:  기준선 결과를 제출 가능한 기여로 정리하고, 신원 억제는 다음 단계로 승인할 것인가?',25,C.navy,true,'center');
    footer(s, 9, 'Source: docs/v2/PROTOCOL.md §0, §11; docs/PI5_BENCHMARK.md; docs/v2/HANDOFF.md.');
    notes(s, 'docs/v2/PROTOCOL.md §0 and §11; docs/PI5_BENCHMARK.md; docs/v2/HANDOFF.md.');
  }

  for (let i=0; i<p.slides.items.length; i++) {
    const slide = p.slides.items[i];
    await saveBlob(`${TMP}/slide-${String(i+1).padStart(2,'0')}.png`, await p.export({ slide, format: 'png', scale: 1 }));
    await fs.writeFile(`${TMP}/slide-${String(i+1).padStart(2,'0')}.layout.json`, await (await slide.export({format:'layout'})).text());
  }
  await saveBlob(`${TMP}/deck-montage.webp`, await p.export({ format: 'webp', montage: true, scale: 1 }));
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(OUT);
}
main().catch(err => { console.error(err); process.exitCode = 1; });
