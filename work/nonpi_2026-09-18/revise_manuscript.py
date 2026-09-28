"""Build an auditable revision from the preserved Downloads manuscript."""
from pathlib import Path
import difflib,json
BASE=Path(__file__).resolve().parent
original=(Path.home()/'Downloads/MCE_blink_detection.tex').read_text(encoding='utf8')
s=original
def replace(old,new):
    global s
    if old not in s: raise ValueError('Missing passage: '+old[:90])
    s=s.replace(old,new,1)
def block(start,end,new):
    global s
    a=s.index(start);b=s.index(end,a)
    s=s[:a]+new+'\n\n'+s[b:]
replace(r'''\title{Eye Blink Detection\\
Using Low-Dimensional\\
Embeddings for Real-Time\\
Edge Deployment}''',r'''\title{Compact Embeddings\\
for Eye Blink Detection\\
on Edge Hardware}''')
for old,new in [(r'\jvol{XX}',r'\jvol{}'),(r'\jnum{XX}',r'\jnum{}'),
                (r'\paper{XX}',r'\paper{}'),(r'\jmonth{xxx/xxx}',r'\jmonth{Review draft}'),
                (r'\publisheddate{DD MM YYYY}',r'\publisheddate{}'),
                (r'\currentdate{DD MM YYYY}',r'\currentdate{}'),
                (r'\pubyear{YYYY}',r'\pubyear{}'),(r'\doiinfo{MCE.YYYY.Doi Number}',r'\doiinfo{}')]:
    replace(old,new)
replace(r'\begin{document}',r'''% First-page footer identifies this working copy rather than a published article.
\makeatletter
\def\ps@plain{\let\@evenhead\@empty\let\@oddhead\@empty
\def\@oddfoot{\vbox{\hsize437pt{\rfxfont\bfseries Review draft}\hfill
\rlap{\hspace*{.75pc}\FolioODD}}}\let\@evenfoot\@oddfoot}
\makeatother
\begin{document}''')
replace(r'\begin{tikzpicture}[',r'\resizebox{\columnwidth}{!}{\begin{tikzpicture}[')
replace(r'\end{tikzpicture}',r'\end{tikzpicture}}')
replace('embedding by an encoder whose strides follow the measured eyelid-gap geometry, and a','embedding by a compact convolutional encoder, and a')
replace('temporal head classifies a buffer of these embeddings rather than raw eye images,','temporal head classifies a buffer of these embeddings,')
block('Under a single protocol ---', '% =====================================================================\n\\section{RELATED WORK}',r'''Under a common crop, subject split and temporal head, we quantify the accuracy--cost
trade-off between a compact encoder and an adapted image-CNN reference. Both learned
frontends produce 16-dimensional features and both cache one representation per frame;
the controlled comparison is therefore between backbones. Our contributions are the
paired subject-level analysis of this trade-off, a transparent accounting of model and
pipeline cost, and a Raspberry Pi~5 stage decomposition showing that landmark detection
limits the end-to-end benefit. The stride and embedding-width ablations are exploratory
checks, not evidence of a uniquely optimal architecture.''')
replace('standard dilated 1-D convolutions; the encoder geometry (described under\n\\emph{Encoder Design and Vertical-Resolution Ablation}) and the encode-once buffering\nscheme (described under \\emph{Temporal Modeling on Embeddings}) are the components\ndesigned here.', 'standard dilated 1-D convolutions. The same buffering strategy applies to the image-CNN\nreference; we measure the cost saved by replacing its backbone.')
replace('The measured eyelid gap is $d_{\\mathrm{lid}}/d_{\\mathrm{io}}=0.1201$, so with the crop\ngeometry above its height is\n$(d_{\\mathrm{lid}}/h_{\\mathrm{crop}})H=0.1201W/2.2=8.73$~px. The encoder has four','The encoder has four')
replace('For every event, $m_j\\in\\{0,1\\}$ records whether window position $j$ is valid, so that\n$\\sum_j m_j\\ge 14$ and both pooling operations below are always defined.', 'For every retained training/test event, $m_j\\in\\{0,1\\}$ records whether window position\n$j$ is valid, with $\\sum_jm_j\\ge14$. This is a dataset inclusion rule; a deployment\nsystem must implement its own explicit missing-frame policy.')
replace('window implies a fixed $\\approx$9-frame ($0.3$~s)\nreporting delay.', 'window places its center nine frames ($0.3$~s at 30~fps) before the decision.\nThis nominal alignment offset is not a measured onset-to-alert latency.')
replace('scores; the two EAR baselines are bit-identical across seeds.', 'scores. The EAR rule is deterministic within a fold; EAR + head is a learned,\nseed-dependent model.')
block('window and concatenates the final hidden states,', '\nEAR is computed',r'''window. Both recurrent heads use masked mean and max pooling over all recurrent outputs,
followed by an MLP; neither uses only the final hidden state. They see only the completed
buffer, although BiLSTM uses both directions within it. The 16-D Image-CNN backbone has
471,536 parameters, and the added TCN, BiLSTM and LSTM heads have 4,625, 8,577 and 4,353
parameters, respectively. The max variant has a 470,561-parameter scalar-output backbone;
its smaller projection must not be counted as a difference in temporal-head size.
\emph{Ours} uses the same 4,625-parameter TCN as Image-CNN (+head).

The recurrent accuracy rows were imported from a training handoff summary. Their raw
per-run predictions and trained checkpoints were not available for this revision;
we retain these rows as reported secondary results, not independently reanalysed evidence.''')
replace('Cost covers parameters, MMAC per frame, latency percentiles, peak memory, CPU\nutilization, and idle-subtracted power and energy per frame;', 'Cost covers parameters, MMAC per frame, latency percentiles, peak memory, CPU\nutilization, idle-subtracted rail power and total-active rail energy per frame;')
block('to \\texttt{performance}, each of the six modes', '\n\\begin{table*}',r'''to \texttt{performance}, six modes were replayed without frame-rate pacing, three times
each. Although the runner requested 300~s, the 18 power runs ended at the video boundary
after 98.65--130.39~s and 10,268 frames, achieving 78.75--104.09~fps. Thus these runs
characterize saturated offline throughput, not power at a matched 30~fps camera rate.
The configured PMIC sampling rate was 7.5~Hz; observed sampling was approximately
6.3~Hz after command overhead. The sum of 12 reported rails is provisional: the rail
definition was not independently verified and is not a whole-board input-power measure.
We report mean active-minus-idle power (30~s idle baseline), but energy per frame is
the trapezoidal integral of total active rail power divided by processed frames.
The two quantities must not be interpreted as the same idle-subtracted metric.
No throttling flags were reported in these runs. Separate earlier 300~s latency runs
provide the end-to-end medians quoted below; they are not the 18 power runs.''')
replace('other metrics are 15-run means.', 'other metrics are 15-run means. Recurrent accuracy rows are imported summary values\nwhose raw prediction files were unavailable for this revision.')
block('The 15-run mean ($-0.0019', '\nThe exploratory \\emph{v-drop}',r'''The mean of the 15 paired run differences is $-0.0019\pm0.0024$. Pooling scores from
different fitted models and averaging their AP values are different estimands; their
values must not be interchanged. In a post-hoc margin sensitivity analysis, the primary
pooled interval also supports $\delta=0.01$, but not $\delta=0.005$. The original
$0.02$ margin was informed by reference variability, not independently validated as a
clinically or operationally acceptable loss.

A second post-hoc analysis gives each subject equal weight: average that subject's AP
over its three source runs, then average the 57 subjects. The paired difference is
$-0.0062$ (subject bootstrap 95\% CI $[-0.0114,-0.0023]$). This supports $0.02$ but
not $0.01$, showing that the stronger margin conclusion depends on weighting.
These intervals condition on the saved fits and do not capture full retraining uncertainty.

The recurrent summary rows suggest sensitivity to head choice, but do not support
new paired inference without their raw predictions. A confidence interval excluding
zero cannot be dismissed merely because its effect is smaller than a seed standard
deviation. The Image-CNN backbone accounts for essentially all of the recurrent
variants' static computational cost. Its 6,912-to-64 dense layer alone has 442,432
parameters; the reported parameter ratio is therefore specific to this adapted
reference and does not establish superiority over all lightweight CNNs.''')
block('Against the stronger EAR baseline,', '\n\\subsection{Model Complexity',r'''Against EAR + head, the pooled-prediction advantage is $+0.0151$ overall. The exploratory
analyses include acquisition batch (two groups), glasses (two groups), and brightness,
contrast and sharpness terciles (nine groups). Their unadjusted intervals describe
observed heterogeneity; they are not simultaneous guarantees of robustness. Against
Image-CNN (+head), the available paired subgroup analysis covers acquisition batch and
glasses, plus the overall result. Those intervals meet $\delta=0.02$; corresponding
photometric comparisons were not available. Natural subgroup differences do not replace
controlled tests of illumination, pose or viewing distance.''')
replace('over three repeats. Latencies are ms/frame.', 'over three free-running repeats (98.65--130.39~s each). Latencies are ms/frame.')
replace('Dec. = decision stage; Sub. = Crop+Enc.+Dec.; e2e99', 'Dec. = decision stage; Sub. = sum of stage medians, not a measured subtotal percentile; e2e99')
replace('end-to-end latency; $\\Delta$P(W) = idle-subtracted active power. Lower is better; bold', 'end-to-end latency; $\\Delta$P(W) = active-minus-idle rail power; mJ/frame = total-active\nrail energy. Rail definition is unverified. Lower is better; bold')
replace('latency only, so these timings should not be combined with the accuracy figures of', 'structural latency and power characterization, so these measurements do not demonstrate\nthe runtime/energy of the trained accuracy checkpoints in')
replace('All modes satisfy the 30-FPS budget, with 99th-percentile\nend-to-end latency at 31--40\\% of the $33.3$~ms frame period.', 'The observed 99th percentiles are below the $33.3$~ms frame period, supporting\nprocessing headroom on this replay. This does not establish live-camera deadline\nreliability or a bound on onset-to-alert delay.')
block('Once every mode clears the 30-FPS budget,', '\n\\subsection{Discussion and Limitations}',r'''Under free-running replay, the proposed model draws $2.000$~W above the measured idle
rail sum and uses $46.15$~mJ/frame of total active rail energy, versus
$53.77$--$54.38$~mJ/frame for the image-CNN variants. Different achieved frame rates,
short runs, unverified rails and random-weight comparison graphs prevent a claim of
energy savings at a matched 30~fps workload or an inferred battery lifetime.

The latency decomposition is the more direct system result. From the earlier mean
latencies, even eliminating the 0.868~ms encoder would reduce the 11.300~ms pipeline
by only 7.7\%, assuming all other stages stay fixed. Halving the 8.487~ms landmark stage
would instead save 37.6\% under that same assumption. These are arithmetic upper-bound
scenarios, not measured frontend optimizations.''')
block('Three limitations bound these results.', '\n\\section{CONCLUSION}',r'''The failure on held-out subject U1 remains unresolved: its mean AP is $0.8200$ versus
$0.9433$ for Image-CNN (+head) and $0.9802$ for EAR + head. Existing diagnostic checks
do not establish a causal explanation or exclude all annotation/preprocessing effects.
U1 is retained in every primary analysis; removing it changes the pooled image-control
difference only from $-0.00402$ to $-0.00388$, so it does not explain the full trade-off.

The primary benchmark classifies fixed 19-frame events, not continuous detections.
Its balanced negative sampling and validation accuracy threshold do not determine the
false-alarm rate at deployment prevalence. The nominal 0.3~s center offset is separate
from compute latency and from measured alert delay. The primary width was fixed at
$D=16$ before final comparisons; exploratory $D=8,32,64$ results on two folds exist,
so the width was not wholly unswept. They do not establish a global optimum. The
vertical-stride ablation likewise does not support the proposed geometry as necessary.
Generalization beyond this 57-subject subset needs external evaluation with explicit
coverage and failure reporting; landmark-source shift is not resolved by using the
same crop code.''')
block('We classify blink events on 16-dimensional', '\n\\section{ACKNOWLEDGMENTS}',r'''A compact encoder with a shared temporal head yields a measured accuracy--cost
trade-off on subject-disjoint mEBAL2 events: the pooled AP loss is small but nonzero,
and falls within the prespecified $0.02$ margin at $5.7\times$ fewer parameters and
$2.6\times$ fewer operations than the adapted Image-CNN control. Both frontends use
cached 16-D features. Raspberry Pi~5 replay shows ample compute headroom, while its
dominant landmark stage limits the benefit of encoder compression. Stronger deployment
claims require continuous external evaluation and matched-rate, verified power measurements.''')
replace('L.~Meng, Z.~Fang, and J.~T. Zhou,','L.~Meng, Z.~Fang, J.~T. Zhou, and J.~Yuan,')
summary=json.loads((BASE/'results/external_summary.json').read_text())['summary']
rows=[]
for key,name in [('ours','Ours'),('image_head','Image-CNN (+head)'),('ear_head','EAR + head'),('ear_rule','EAR (rule)')]:
    r=summary[key]
    rows.append(f"{name} & {r['window_AP']['mean']:.3f} & {r['event_recall']['mean']:.3f} & {r['event_F1']['mean']:.3f} & {r['false_events_per_minute']['mean']:.2f} \\\\")
external=r'''\subsection{External Continuous-Video Pilot}
We additionally evaluate the locally available original EyeBlink8 video~9
\cite{fogelton2016,eyeblink8data}: one person, 41 annotated blink IDs. This is one of
eight benchmark videos and had previously been used without accuracy labels for timing.
The container and timestamp file list 5,183 frames, but 5,134 decode, and the annotation
ends at decoded frame 5,133 (zero-based). We discard the 49 unmatched tail timestamps.
Nearest-frame resampling using capture times gives 5,133 positions at 30~Hz, with 174
duplicates and 175 unused native frames. MediaPipe crop coverage is 99.75\%; all
annotated evaluation windows meet the 14-of-19 validity rule. Sampled annotation/eye-corner
alignment and checkpoint reproduction were checked as implementation audits.

All 15 source-trained checkpoints and their source-validation thresholds are frozen.
No external training, threshold selection or best-run selection is performed. The
window diagnostic uses 41 blink-centered windows and 210 deterministic, non-overlapping
negative windows without blink overlap. Continuous evaluation thresholds every trailing
window and groups consecutive positives, without tuned hysteresis. Predicted intervals
are aligned to the window center and matched one-to-one to overlapping blink-ID
intervals. All 41 blinks are included over 2.80 annotated minutes. This bilateral-ID
protocol is distinct from the dataset authors' per-eye completeness evaluation.

\begin{table}[!t]
\caption{Descriptive external pilot: one EyeBlink8 video. Means over 15 source fits;
these fits are not 15 independent external subjects. AP refers to sampled windows;
R, F1 and false events/minute (FA/min) refer to continuous detection.}
\label{table_external}
\centering\small\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\columnwidth}{@{}l@{\extracolsep{\fill}}cccc@{}}
\hline
Method & AP & R & F1 & FA/min\\
\hline
'''+ '\n'.join(rows)+r'''
\hline
\end{tabular*}
\end{table}

The proposed model's continuous F1 varies from 0.726 to 0.976 across source fits
(mean 0.889, SD 0.063), despite a mean window AP of 0.999. It produces 3.81 unmatched
detections/minute on average (SD 2.57), and its negative-frame positive rate is 0.63\%.
Requiring temporal intersection-over-union of at least 0.1 gives the same proposed-model
F1 here. The mean of run-wise median onset-to-first-decision offsets is 0.307~s, computed
from capture timestamps and excluding processing, display and queue delays. This pilot
demonstrates a measurable classification-to-continuous-detection gap; its single subject
cannot establish cross-dataset superiority or a population confidence interval.

'''
replace(r'\subsection{Discussion and Limitations}',external+r'\subsection{Discussion and Limitations}')
replace('Generalization beyond this 57-subject subset needs external evaluation with explicit\ncoverage and failure reporting;', 'The one-subject external pilot does not establish generalization beyond the\n57-subject source subset;')
replace(r'\begin{thebibliography}{10}',r'''% Compact reference leading prevents a nearly empty final page; font size is unchanged.
\renewcommand{\bibliofont}{\fontfamily{\sfdefault}\fontsize{8}{11.5}\selectfont\raggedright}
\begin{thebibliography}{12}''')
replace(r'\end{thebibliography}',r'''\bibitem{fogelton2016}
A.~Fogelton and W.~Benesova, ``Eye blink detection based on motion vectors analysis,''
\emph{Comput. Vis. Image Underst.}, vol.~148, pp.~23--33, 2016.

\bibitem{eyeblink8data}
Blinking Matters, ``Research: EyeBlink8 dataset and annotations,'' accessed Sep.~18, 2026.
[Online]. Available: \url{https://www.blinkingmatters.com/research}

\end{thebibliography}''')
p=BASE/'manuscript/MCE_blink_detection.tex';p.write_text(s,encoding='utf8')
(BASE/'manuscript_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='original_Downloads.tex',tofile='revised_workspace.tex')),encoding='utf8')
print('Revised',p)
