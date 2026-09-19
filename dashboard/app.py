from pathlib import Path
from io import BytesIO
import html, joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import skew, kurtosis

st.set_page_config(page_title='UAV Edge-AI Fault Diagnosis', page_icon='🚁', layout='wide', initial_sidebar_state='expanded')
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
ASSET_DIR = Path(__file__).resolve().parent / 'assets'
STM32_BUILD_IMG = ASSET_DIR / 'stm32_release_build.png'
RENODE_IMG = ASSET_DIR / 'renode_validation.png'
FS, WINDOW_SIZE, FFT_SIZE = 1023.5414534523528, 500, 512
FLASH_USED, STATIC_RAM_USED = 18976, 10152
STM_FLASH_TOTAL, STM_RAM_TOTAL = 512*1024, 128*1024
CLASS_NAMES = {0:'Healthy',1:'Damaged Bottom Right Blade',2:'Damaged Top Right Blade',3:'Unbalanced Bottom Right Blade',4:'Unbalanced Top Right Blade'}
SHORT_NAMES = {0:'Healthy',1:'Damaged BR',2:'Damaged TR',3:'Unbalanced BR',4:'Unbalanced TR'}
FAULT_TEXT = {0:'Normal rotor condition; no blade damage or imbalance is indicated.',1:'Blade damage is present on the bottom-right rotor.',2:'Blade damage is present on the top-right rotor.',3:'Mass imbalance / abnormal vibration is present on the bottom-right rotor.',4:'Mass imbalance / abnormal vibration is present on the top-right rotor.'}
FEATURE_NAMES = ['x_mean','x_std','x_rms','x_ptp','x_crest','x_skew','x_kurtosis','x_spec_centroid','y_mean','y_std','y_rms','y_ptp','y_crest','y_skew','y_kurtosis','y_spec_centroid']

st.markdown("""
<style>
.block-container{max-width:1480px;padding-top:1.35rem;padding-bottom:.9rem}
h1{font-size:2.15rem!important;margin:.15rem 0 .35rem 0!important;letter-spacing:-.5px}
h2{font-size:1.45rem!important;margin:.25rem 0 .45rem 0!important} h3{font-size:1.08rem!important;margin:.2rem 0 .35rem 0!important}
p,li,div{font-size:.95rem}[data-testid='stSidebar']{min-width:292px;max-width:312px}[data-testid='stSidebar']>div:first-child{padding-top:.15rem!important}[data-testid='stSidebar'] [data-testid='stSidebarContent']{padding-top:0!important}[data-testid='stSidebar'] .block-container{padding-top:.2rem!important}[data-testid='stSidebar'] h1{margin-top:0!important;margin-bottom:.35rem!important;padding-top:0!important;font-size:1.9rem!important}[data-testid='stSidebar'] hr{margin:.55rem 0!important}
[data-testid='stAppDeployButton'],#MainMenu,footer{display:none!important}
.hero-subtitle{color:#9daecb;font-size:.96rem;margin:.05rem 0 .7rem 0}.pipeline{display:flex;flex-wrap:wrap;align-items:center;gap:5px;margin:3px 0 12px 0}.pipeline-box{background:rgba(20,33,58,.88);border:1px solid rgba(105,145,220,.28);padding:5px 9px;border-radius:9px;font-size:.8rem;white-space:nowrap}.pipeline-arrow{color:#7791ba;font-size:.85rem}
.kpi{min-height:92px;background:linear-gradient(145deg,rgba(20,34,60,.96),rgba(10,18,34,.96));border:1px solid rgba(105,145,220,.28);border-radius:14px;padding:11px 13px}.kpi-title{color:#9daecb;font-size:.7rem;text-transform:uppercase;letter-spacing:.55px}.kpi-value{font-size:1.17rem;font-weight:650;line-height:1.18;margin-top:6px}.kpi-detail{color:#8c9bb6;font-size:.75rem;margin-top:4px}
.info-card,.compact-card{background:rgba(20,36,63,.82);border:1px solid rgba(104,146,215,.28);border-radius:14px;padding:13px 14px;line-height:1.42}.status-grid,.contract-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:8px}.status-box,.contract-box{background:rgba(14,27,46,.78);border:1px solid rgba(100,140,200,.2);border-radius:10px;padding:8px 9px}.status-label,.contract-key{color:#91a3bf;font-size:.68rem;text-transform:uppercase;letter-spacing:.4px}.status-value,.contract-value{font-size:.88rem;font-weight:650;margin-top:2px}.result-good{color:#6ee7a8;font-weight:650}.result-warn{color:#ffd166;font-weight:650}.badge{display:inline-block;padding:3px 8px;border-radius:8px;font-size:.76rem;border:1px solid rgba(48,185,119,.35);background:rgba(48,185,119,.11);color:#8df0b7}
.diagram{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:12px;align-items:center;margin:14px 0 20px 0}.diagram-box{background:rgba(15,28,48,.8);border:1px solid rgba(105,145,220,.24);border-radius:12px;padding:10px 11px;text-align:center;min-height:72px;display:flex;flex-direction:column;justify-content:center}.diagram-title{font-weight:700;font-size:.88rem}.diagram-sub{color:#8fa2c1;font-size:.72rem;margin-top:3px;line-height:1.25}.diagram-arrow{color:#7ca6e8;font-size:1.2rem;text-align:center}.branch-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:10px 0 20px 0}.branch{background:rgba(16,29,50,.78);border:1px solid rgba(105,145,220,.22);border-radius:12px;padding:10px}.branch-title{font-size:.86rem;font-weight:700;margin-bottom:4px}.branch-sub{color:#8fa2c1;font-size:.74rem;line-height:1.3}
.step-row{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:10px 0 18px 0}.step-card{min-height:82px;background:rgba(16,28,48,.76);border:1px solid rgba(100,140,205,.2);border-radius:10px;padding:8px 9px}.step-num{color:#8fb5ff;font-size:.7rem;font-weight:700}.step-title{font-size:.85rem;font-weight:650;margin-top:3px}.step-sub{color:#8c9bb6;font-size:.69rem;margin-top:3px;line-height:1.25}.caption-box{color:#92a1ba;font-size:.8rem;line-height:1.35}
div[data-testid='stMetric']{background:rgba(18,30,52,.65);border:1px solid rgba(105,145,220,.22);padding:9px 11px;border-radius:12px}div[data-testid='stMetricValue']{font-size:1.25rem}button[data-baseweb='tab']{padding-top:.4rem;padding-bottom:.4rem}
@media(max-width:900px){.diagram{grid-template-columns:1fr}.diagram-arrow{transform:rotate(90deg)}.branch-grid,.status-grid,.contract-grid{grid-template-columns:1fr}.step-row{grid-template-columns:1fr 1fr}}
</style>
""", unsafe_allow_html=True)

def find_dataset_file(class_id):
    target = CLASS_NAMES[class_id].lower()
    if not DATA_DIR.exists(): return None
    return next((p for p in DATA_DIR.rglob('*.xlsx') if target in p.stem.lower()), None)

@st.cache_data(show_spinner=False)
def load_excel(path_string):
    df = pd.read_excel(path_string); numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] < 3: raise ValueError('Expected numeric time, X and Y acceleration columns.')
    cols = list(numeric.columns)
    def axis(name, fallback):
        for c in cols:
            s = str(c).lower().replace('_',' ')
            if s == name or f'{name} axis' in s or f'acceleration {name}' in s or f'{name} acceleration' in s:
                return numeric[c].to_numpy(dtype=np.float64)
        return numeric.iloc[:, fallback].to_numpy(dtype=np.float64)
    return axis('x',1), axis('y',2)

@st.cache_data(show_spinner=False)
def load_excel_bytes(file_bytes):
    df = pd.read_excel(BytesIO(file_bytes))
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] < 3:
        raise ValueError('Expected numeric time, X and Y acceleration columns.')
    cols = list(numeric.columns)
    def axis(name, fallback):
        for c in cols:
            s = str(c).lower().replace('_',' ')
            if s == name or f'{name} axis' in s or f'acceleration {name}' in s or f'{name} acceleration' in s:
                return numeric[c].to_numpy(dtype=np.float64)
        return numeric.iloc[:, fallback].to_numpy(dtype=np.float64)
    return axis('x',1), axis('y',2)

def spectral_centroid(signal):
    centered = np.asarray(signal,dtype=np.float64)-np.mean(signal); padded=np.zeros(FFT_SIZE); padded[:WINDOW_SIZE]=centered[:WINDOW_SIZE]
    power=np.abs(np.fft.rfft(padded))**2; freqs=np.fft.rfftfreq(FFT_SIZE,d=1/FS); total=np.sum(power)
    return 0.0 if total<=1e-20 else float(np.sum(freqs*power)/total)

def axis_features(signal):
    s=np.asarray(signal,dtype=np.float64); mean=float(np.mean(s)); std=float(np.std(s,ddof=1)); rms=float(np.sqrt(np.mean(s**2))); ptp=float(np.ptp(s)); peak=float(np.max(np.abs(s)))
    crest=peak/rms if rms>1e-12 else 0.0; sk=float(skew(s,bias=False)); ku=float(kurtosis(s,fisher=True,bias=False)); sk=0.0 if not np.isfinite(sk) else sk; ku=0.0 if not np.isfinite(ku) else ku
    return [mean,std,rms,ptp,crest,sk,ku,spectral_centroid(s)]

def extract_features(x,y): return pd.DataFrame([axis_features(x)+axis_features(y)],columns=FEATURE_NAMES)
def fft_curve(signal):
    centered=np.asarray(signal,dtype=np.float64)-np.mean(signal); padded=np.zeros(FFT_SIZE); padded[:WINDOW_SIZE]=centered; spec=np.fft.rfft(padded)
    return np.fft.rfftfreq(FFT_SIZE,d=1/FS),np.abs(spec)

def unwrap_model(obj):
    if hasattr(obj,'predict'): return obj
    if isinstance(obj,dict):
        for k in ('model','classifier','estimator','adaboost','final_model'):
            if hasattr(obj.get(k),'predict'): return obj[k]
    return None

def is_adaboost(model):
    if model is None: return False
    if 'adaboost' in model.__class__.__name__.lower(): return True
    return hasattr(model,'named_steps') and any('adaboost' in s.__class__.__name__.lower() for s in model.named_steps.values())

@st.cache_resource(show_spinner=False)
def load_joblib_cached(path_string): return joblib.load(path_string)

def load_final_adaboost():
    candidates=[]
    for loc in (ROOT/'models',ROOT/'checkpoints',ROOT/'artifacts',ROOT):
        if loc.exists():
            for ext in ('*.joblib','*.pkl'): candidates.extend(loc.rglob(ext))
    def score(p):
        n=p.name.lower(); return 30*('ada' in n)+15*('final' in n)+10*(('fft512' in n) or ('512' in n))+8*('embedded' in n)+4*('xy' in n)
    for p in sorted(set(candidates),key=score,reverse=True):
        try:
            model=unwrap_model(load_joblib_cached(str(p)))
            if is_adaboost(model): return model
        except Exception: pass
    return None

def run_model(model,feature_df):
    X=feature_df.copy()
    if hasattr(model,'feature_names_in_'):
        req=list(model.feature_names_in_)
        if set(req)==set(X.columns): X=X[req]
    try: pred=int(model.predict(X)[0])
    except Exception: pred=int(model.predict(X.to_numpy(dtype=np.float32))[0])
    classes=proba=None
    if hasattr(model,'predict_proba'):
        try: proba=model.predict_proba(X)[0]
        except Exception: proba=model.predict_proba(X.to_numpy(dtype=np.float32))[0]
        classes=getattr(model,'classes_',np.arange(len(proba)))
    return pred,classes,proba

def add_box(fig,cx,cy,cz,sx,sy,sz,color,name=None,showlegend=False):
    x=np.array([-1,1,1,-1,-1,1,1,-1])*sx/2+cx; y=np.array([-1,-1,1,1,-1,-1,1,1])*sy/2+cy; z=np.array([-1,-1,-1,-1,1,1,1,1])*sz/2+cz
    faces=[(0,1,2),(0,2,3),(4,6,5),(4,7,6),(0,4,5),(0,5,1),(1,5,6),(1,6,2),(2,6,7),(2,7,3),(3,7,4),(3,4,0)]
    fig.add_trace(go.Mesh3d(x=x,y=y,z=z,i=[f[0] for f in faces],j=[f[1] for f in faces],k=[f[2] for f in faces],color=color,opacity=.98,name=name,showlegend=showlegend,flatshading=True))

def add_line(fig,p1,p2,color,width=7): fig.add_trace(go.Scatter3d(x=[p1[0],p2[0]],y=[p1[1],p2[1]],z=[p1[2],p2[2]],mode='lines',line=dict(color=color,width=width),showlegend=False,hoverinfo='skip'))
def add_ring(fig,mx,my,z,radius,color,width=2):
    t=np.linspace(0,2*np.pi,90); fig.add_trace(go.Scatter3d(x=mx+radius*np.cos(t),y=my+radius*np.sin(t),z=np.full_like(t,z),mode='lines',line=dict(color=color,width=width),showlegend=False,hoverinfo='skip'))

def add_propeller(fig,name,mx,my,mz,fault_motor,class_id,angle):
    damaged=name==fault_motor and class_id in (1,2); unbalanced=name==fault_motor and class_id in (3,4); color='#ff5a67' if damaged else '#ffb347' if unbalanced else '#7bdcff'
    add_ring(fig,mx,my,mz+.05,.34,color,2); c,s=np.cos(angle),np.sin(angle); c2,s2=np.cos(angle+np.pi/2),np.sin(angle+np.pi/2)
    if damaged:
        fig.add_trace(go.Scatter3d(x=[mx-.31*c,mx+.10*c],y=[my-.31*s,my+.10*s],z=[mz+.055,mz+.055],mode='lines',line=dict(color=color,width=9),showlegend=False))
        fig.add_trace(go.Scatter3d(x=[mx+.20*c,mx+.31*c+.06],y=[my+.20*s,my+.31*s+.05],z=[mz+.04,mz-.02],mode='lines',line=dict(color='#ff3131',width=8),showlegend=False))
    elif unbalanced:
        fig.add_trace(go.Scatter3d(x=[mx-.23*c,mx+.37*c],y=[my-.23*s,my+.37*s],z=[mz+.055,mz+.055],mode='lines',line=dict(color=color,width=9),showlegend=False)); add_ring(fig,mx+.025,my-.025,mz+.07,.40,'rgba(255,179,71,.65)',2); add_ring(fig,mx-.02,my+.02,mz+.075,.46,'rgba(255,90,60,.30)',2)
    else:
        fig.add_trace(go.Scatter3d(x=[mx-.31*c,mx+.31*c],y=[my-.31*s,my+.31*s],z=[mz+.055,mz+.055],mode='lines',line=dict(color=color,width=8),showlegend=False))
    fig.add_trace(go.Scatter3d(x=[mx-.27*c2,mx+.27*c2],y=[my-.27*s2,my+.27*s2],z=[mz+.052,mz+.052],mode='lines',line=dict(color=color,width=7),showlegend=False))

def create_uav_figure(class_id):
    fig=go.Figure(); fault_motor={1:'Bottom Right',2:'Top Right',3:'Bottom Right',4:'Top Right'}.get(class_id)
    motors={'Top Left':(-1,1,.02),'Top Right':(1,1,.02),'Bottom Left':(-1,-1,.02),'Bottom Right':(1,-1,.02)}; arm={'Top Left':'#58d0c3','Top Right':'#e6edf7','Bottom Left':'#e6edf7','Bottom Right':'#58d0c3'}
    add_box(fig,0,0,.03,.72,.52,.18,'#263445'); add_box(fig,0,0,.15,.54,.38,.08,'#3b5069'); add_box(fig,0,-.02,-.12,.48,.32,.12,'#161d28')
    add_box(fig,0,0,.245,.46,.27,.045,'#2a9d63','STM32F446RE',True); add_box(fig,0,0,.277,.13,.13,.025,'#101820'); add_box(fig,.19,0,.272,.09,.12,.035,'#aeb7c2')
    py=np.linspace(-.095,.095,8); fig.add_trace(go.Scatter3d(x=np.r_[np.full(8,-.19),np.full(8,.19)],y=np.r_[py,py],z=np.full(16,.283),mode='markers',marker=dict(size=3,color='#f6c453'),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[.34],mode='text',text=['STM32F446RE'],textfont=dict(size=11,color='#74f0a7'),showlegend=False,hoverinfo='skip'))
    angles={'Top Left':.3,'Top Right':1.1,'Bottom Left':1.1,'Bottom Right':.3}
    for name,(mx,my,mz) in motors.items():
        add_line(fig,(0,0,.04),(mx,my,mz),arm[name],9); fig.add_trace(go.Scatter3d(x=[mx],y=[my],z=[mz],mode='markers',marker=dict(size=10,color='#404c5c',line=dict(width=2,color=arm[name])),showlegend=False,hovertext=name,hoverinfo='text')); add_propeller(fig,name,mx,my,mz,fault_motor,class_id,angles[name])
    for xx in (-.36,.36): add_line(fig,(xx,-.27,-.12),(xx,-.27,-.36),'#8b98a9',5); add_line(fig,(xx,.27,-.12),(xx,.27,-.36),'#8b98a9',5); add_line(fig,(xx,-.36,-.36),(xx,.36,-.36),'#8b98a9',6)
    if fault_motor:
        mx,my,mz=motors[fault_motor]; label='DAMAGED BLADE' if class_id in (1,2) else 'UNBALANCED ROTOR'; fig.add_trace(go.Scatter3d(x=[mx],y=[my],z=[mz+.52],mode='markers+text',marker=dict(size=5,color='#ff5a67'),text=[label],textposition='top center',textfont=dict(size=12,color='#ff737d'),showlegend=False))
    fig.update_layout(height=480,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',scene=dict(bgcolor='rgba(0,0,0,0)',xaxis=dict(visible=False,range=[-1.6,1.6]),yaxis=dict(visible=False,range=[-1.6,1.6]),zaxis=dict(visible=False,range=[-.65,.8]),aspectmode='manual',aspectratio=dict(x=1,y=1,z=.55),camera=dict(eye=dict(x=1.45,y=-1.55,z=1.12))),legend=dict(orientation='h',x=.02,y=.98,bgcolor='rgba(0,0,0,0)'))
    return fig

# ---------------- SIDEBAR ----------------
st.sidebar.title('Scenario Control')
st.sidebar.caption('Replay benchmark data or evaluate a new vibration recording through the same frozen inference pipeline.')

source_mode = st.sidebar.radio(
    'Input source',
    ['Recorded benchmark', 'Upload new .xlsx'],
)

known_condition = source_mode == 'Recorded benchmark'
uploaded_name = None

if known_condition:
    selected_class = st.sidebar.selectbox(
        'Operating condition',
        list(CLASS_NAMES),
        format_func=lambda v: CLASS_NAMES[v],
    )
    dataset_file = find_dataset_file(selected_class)
    if dataset_file is None:
        st.error(f'Dataset for {CLASS_NAMES[selected_class]} was not found.')
        st.stop()
    x_all, y_all = load_excel(str(dataset_file))
    source_detail = CLASS_NAMES[selected_class]
else:
    selected_class = None
    uploaded = st.sidebar.file_uploader(
        'Upload vibration file',
        type=['xlsx'],
        help='Expected numeric columns: time, X acceleration and Y acceleration. Additional columns are allowed.',
    )
    if uploaded is None:
        st.info(
            'Upload an .xlsx vibration recording from the sidebar to run inference on new data. '
            'The file is classified by the frozen model; it is not automatically added to training.'
        )
        st.stop()
    uploaded_name = uploaded.name
    try:
        x_all, y_all = load_excel_bytes(uploaded.getvalue())
    except Exception as exc:
        st.error(f'Could not read uploaded file: {exc}')
        st.stop()
    source_detail = 'Uploaded vibration recording'

max_window = max(0, len(x_all)//WINDOW_SIZE - 1)
if len(x_all) < WINDOW_SIZE:
    st.error(f'At least {WINDOW_SIZE} samples are required; this file contains {len(x_all)} usable samples.')
    st.stop()

segment_index = st.sidebar.number_input(
    'Signal window',
    min_value=0,
    max_value=max_window,
    value=0,
    step=1,
)
start = int(segment_index)*WINDOW_SIZE
x, y = x_all[start:start+WINDOW_SIZE], y_all[start:start+WINDOW_SIZE]

st.sidebar.divider()
st.sidebar.markdown('### Runtime specification')
st.sidebar.markdown(
    f"""<div class='contract-grid'>
    <div class='contract-box'><div class='contract-key'>Sampling</div><div class='contract-value'>{FS:.1f} Hz</div></div>
    <div class='contract-box'><div class='contract-key'>Window</div><div class='contract-value'>{WINDOW_SIZE}</div></div>
    <div class='contract-box'><div class='contract-key'>FFT</div><div class='contract-value'>{FFT_SIZE}-pt</div></div>
    <div class='contract-box'><div class='contract-key'>Axes</div><div class='contract-value'>X + Y</div></div>
    <div class='contract-box'><div class='contract-key'>Features</div><div class='contract-value'>16</div></div>
    <div class='contract-box'><div class='contract-key'>Classes</div><div class='contract-value'>5</div></div>
    </div>""",
    unsafe_allow_html=True,
)

features = extract_features(x,y)
model = load_final_adaboost()
if model is None:
    st.error('Reference AdaBoost model not found. Place the final .joblib/.pkl artifact under models/, checkpoints/ or artifacts/.')
    st.stop()

prediction, probability_classes, probabilities = run_model(model,features)
agreement = (prediction == selected_class) if known_condition else None
visual_class = selected_class if known_condition else prediction

# ---------------- HEADER ----------------
st.title('UAV Propeller Edge-AI Fault Diagnosis')
st.markdown("<div class='hero-subtitle'>Vibration-based rotor health monitoring with an embedded-compatible ML pipeline for STM32F446RE.</div>",unsafe_allow_html=True)
st.markdown("""<div class='pipeline'><div class='pipeline-box'>Accelerometer</div><div class='pipeline-arrow'>→</div><div class='pipeline-box'>500-sample window</div><div class='pipeline-arrow'>→</div><div class='pipeline-box'>DSP + 512 FFT</div><div class='pipeline-arrow'>→</div><div class='pipeline-box'>16 features</div><div class='pipeline-arrow'>→</div><div class='pipeline-box'>AdaBoost</div><div class='pipeline-arrow'>→</div><div class='pipeline-box'>STM32F446RE</div></div>""",unsafe_allow_html=True)
summary=[
    ('RECORDED CONDITION' if known_condition else 'INPUT SOURCE', SHORT_NAMES[selected_class] if known_condition else 'New data', CLASS_NAMES[selected_class] if known_condition else (uploaded_name or 'Uploaded recording')),
    ('REFERENCE MODEL', SHORT_NAMES[prediction], (('Matches label' if agreement else 'Model disagreement') if known_condition else 'Inference on unseen input')),
    ('EMBEDDED-C TWIN', SHORT_NAMES[prediction], 'Parity-backed equivalent output'),
    ('STM32 FOOTPRINT', f'{FLASH_USED/1024:.1f} KiB Flash', f'{STATIC_RAM_USED/1024:.1f} KiB static RAM'),
]
for col,(title,value,detail) in zip(st.columns(4),summary):
    with col: st.markdown(f"<div class='kpi'><div class='kpi-title'>{html.escape(title)}</div><div class='kpi-value'>{html.escape(value)}</div><div class='kpi-detail'>{html.escape(detail)}</div></div>",unsafe_allow_html=True)

overview_tab,signal_tab,feature_tab,embedded_tab,architecture_tab=st.tabs(['Overview','Signal & FFT','Features','Embedded','Architecture'])

with overview_tab:
    left,right=st.columns([1.25,.85],gap='large')
    with left:
        st.subheader('3D UAV Digital Twin')
        st.plotly_chart(create_uav_figure(visual_class),use_container_width=True,config={'displaylogo':False,'scrollZoom':True})
        st.caption(
            'The highlighted rotor reflects the recorded condition for benchmark data, or the model-predicted condition for an uploaded recording. '
            'The green module mounted on the frame represents the STM32F446RE deployment target.'
        )
    with right:
        st.subheader('Diagnosis')
        if known_condition:
            agreement_text='Prediction agrees with recorded condition' if agreement else 'Prediction differs from recorded condition'
            agreement_class='result-good' if agreement else 'result-warn'
        else:
            agreement_text='Prediction generated for uploaded data'
            agreement_class='result-good'
        interpreted_class = selected_class if known_condition else prediction
        st.markdown(
            f"""<div class='info-card'>
            <b>{html.escape(CLASS_NAMES[prediction])}</b><br>
            <span class='{agreement_class}'>{html.escape(agreement_text)}</span>
            <div class='status-grid'>
            <div class='status-box'><div class='status-label'>Interpreted state</div><div class='status-value'>{html.escape(FAULT_TEXT[interpreted_class])}</div></div>
            <div class='status-box'><div class='status-label'>Window duration</div><div class='status-value'>{WINDOW_SIZE/FS:.4f} s</div></div>
            <div class='status-box'><div class='status-label'>Embedded validation</div><div class='status-value'>C parity + Renode verified</div></div>
            <div class='status-box'><div class='status-label'>Deployment target</div><div class='status-value'>STM32F446RE / Cortex-M4F</div></div>
            </div><br><span class='badge'>End-to-end Edge-AI workflow validated</span>
            </div>""",
            unsafe_allow_html=True,
        )
        if not known_condition:
            st.caption('Uploaded data is used for inference only. The frozen model is not retrained or updated automatically.')
        if probabilities is not None:
            prob_df=pd.DataFrame({'Class':[SHORT_NAMES.get(int(c),str(c)) for c in probability_classes],'Score':probabilities}).sort_values('Score')
            fig=go.Figure(go.Bar(x=prob_df['Score'],y=prob_df['Class'],orientation='h',text=[f'{v:.1%}' for v in prob_df['Score']],textposition='auto'))
            fig.update_layout(title='Model class distribution',height=220,margin=dict(l=5,r=5,t=35,b=5),xaxis=dict(range=[0,max(.35,float(prob_df['Score'].max())*1.15)],tickformat='.0%'),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False})

with signal_tab:
    c1,c2=st.columns(2,gap='large'); time_axis=np.arange(WINDOW_SIZE)/FS
    with c1:
        st.subheader('Recorded vibration'); fig=go.Figure(); fig.add_trace(go.Scatter(x=time_axis,y=x,mode='lines',name='X-axis')); fig.add_trace(go.Scatter(x=time_axis,y=y,mode='lines',name='Y-axis')); fig.update_layout(height=420,xaxis_title='Time (s)',yaxis_title='Acceleration',margin=dict(l=15,r=15,t=15,b=15),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',legend=dict(orientation='h',y=-.18)); st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False})
    with c2:
        st.subheader('512-point frequency spectrum'); fx,mx=fft_curve(x); fy,my=fft_curve(y); fig=go.Figure(); fig.add_trace(go.Scatter(x=fx,y=mx,mode='lines',name='X spectrum')); fig.add_trace(go.Scatter(x=fy,y=my,mode='lines',name='Y spectrum')); fig.update_layout(height=420,xaxis_title='Frequency (Hz)',yaxis_title='Magnitude',margin=dict(l=15,r=15,t=15,b=15),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',legend=dict(orientation='h',y=-.18)); st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False})
    st.markdown("<div class='caption-box'>Time-domain features use the 500 recorded samples directly. For spectral analysis, the mean is removed and the signal is zero-padded from 500 to 512 samples before the FFT.</div>",unsafe_allow_html=True)

with feature_tab:
    st.subheader('Embedded-compatible feature vector'); st.caption('The exact 16-feature contract consumed by the reference model and reproduced in the STM32 C pipeline.')
    x_table=pd.DataFrame({'X-axis feature':[n.replace('x_','') for n in FEATURE_NAMES[:8]],'Value':[f'{v:.5f}' for v in features.iloc[0,:8]]}); y_table=pd.DataFrame({'Y-axis feature':[n.replace('y_','') for n in FEATURE_NAMES[8:]],'Value':[f'{v:.5f}' for v in features.iloc[0,8:]]})
    c1,c2,c3=st.columns([1,1,.9],gap='large')
    with c1: st.markdown('**X-axis features**'); st.dataframe(x_table,use_container_width=True,hide_index=True,height=325)
    with c2: st.markdown('**Y-axis features**'); st.dataframe(y_table,use_container_width=True,hide_index=True,height=325)
    with c3: st.markdown("""<div class='compact-card'><b>Feature contract</b><div class='status-grid'><div class='status-box'><div class='status-label'>Time features</div><div class='status-value'>7 / axis</div></div><div class='status-box'><div class='status-label'>Spectral</div><div class='status-value'>1 / axis</div></div><div class='status-box'><div class='status-label'>Total</div><div class='status-value'>16 features</div></div><div class='status-box'><div class='status-label'>Classifier</div><div class='status-value'>50-tree AdaBoost</div></div></div><br><b>Why X + Y?</b><br>Axis-ablation experiments showed that the two-axis configuration retained most diagnostic information while reducing embedded processing cost.</div>""",unsafe_allow_html=True)

with embedded_tab:
    st.subheader('STM32F446RE deployment validation'); v1,v2,v3,v4=st.columns(4); v1.metric('AdaBoost trees','50'); v2.metric('Classifier parity','798 / 798'); v3.metric('Raw pipeline parity','15 / 15'); v4.metric('Renode self-test','15 / 15')
    left,right=st.columns(2,gap='large')
    with left:
        st.markdown('### Resource footprint'); f_pct=FLASH_USED/STM_FLASH_TOTAL; r_pct=STATIC_RAM_USED/STM_RAM_TOTAL; a,b=st.columns(2); a.metric('Flash',f'{FLASH_USED/1024:.1f} KiB',f'{f_pct*100:.1f}% of 512 KiB'); b.metric('Static RAM',f'{STATIC_RAM_USED/1024:.1f} KiB',f'{r_pct*100:.1f}% of 128 KiB'); st.progress(min(f_pct,1.0)); st.caption('Flash utilization'); st.progress(min(r_pct,1.0)); st.caption('Static RAM utilization')
    with right:
        st.markdown('### Model trade-off'); comparison=pd.DataFrame({'Model':['AdaBoost','Compact XGBoost'],'Holdout Macro-F1':['0.9081','0.9099'],'Min class recall':['0.7170','0.6918'],'Trees':[50,375]}); st.dataframe(comparison,use_container_width=True,hide_index=True,height=142); st.markdown("<div class='caption-box'>XGBoost gained only ~0.18 percentage points on the temporal holdout while requiring 7.5× more trees. AdaBoost therefore offered the stronger embedded resource/performance trade-off.</div>",unsafe_allow_html=True)
    st.markdown('### Validation chain'); st.markdown("""<div class='step-row'><div class='step-card'><div class='step-num'>01</div><div class='step-title'>Python reference</div><div class='step-sub'>Frozen preprocessing + AdaBoost</div></div><div class='step-card'><div class='step-num'>02</div><div class='step-title'>C export</div><div class='step-sub'>Classifier reproduced in embedded C</div></div><div class='step-card'><div class='step-num'>03</div><div class='step-title'>Parity</div><div class='step-sub'>798/798 + 15/15 validation</div></div><div class='step-card'><div class='step-num'>04</div><div class='step-title'>STM32 build</div><div class='step-sub'>Optimized Cortex-M4F Release firmware</div></div><div class='step-card'><div class='step-num'>05</div><div class='step-title'>Renode</div><div class='step-sub'>15/15 firmware self-tests passed</div></div></div>""",unsafe_allow_html=True)
    st.markdown('### Deployment evidence')
    e1,e2=st.columns(2,gap='large')
    with e1:
        with st.expander('STM32CubeIDE — Release build evidence'):
            st.markdown('**Result:** optimized Release firmware built with **0 errors / 0 warnings**.  \n**Size:** `text=18888`, `data=88`, `bss=10064` bytes.')
            if STM32_BUILD_IMG.exists():
                st.image(str(STM32_BUILD_IMG),caption='STM32CubeIDE optimized Release build',use_container_width=True)
            else:
                st.code('text   data   bss    dec\n18888    88  10064  29040\n\nBuild Finished. 0 errors, 0 warnings.',language='text')
    with e2:
        with st.expander('Renode — firmware execution evidence'):
            st.markdown('**Result:** firmware self-test completed and **15 / 15** test windows passed.  \n`g_uav_self_test_done = 1` and `g_uav_self_test_passed = 0x0000000F`.')
            if RENODE_IMG.exists():
                st.image(str(RENODE_IMG),caption='Renode STM32F4/Cortex-M4 firmware self-test',use_container_width=True)
            else:
                st.code('g_uav_self_test_done   = 0x00000001\ng_uav_self_test_passed = 0x0000000F  # 15',language='text')

with architecture_tab:
    st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
    st.subheader('System architecture')
    st.caption('The visualization layer and embedded firmware are two execution targets of the same frozen inference contract.')
    st.markdown("""<div class='diagram'><div class='diagram-box'><div class='diagram-title'>UAV vibration data</div><div class='diagram-sub'>Recorded X/Y accelerometer signals</div></div><div class='diagram-arrow'>→</div><div class='diagram-box'><div class='diagram-title'>DSP preprocessing</div><div class='diagram-sub'>500 samples • statistics • mean removal • zero-pad 512 • FFT</div></div><div class='diagram-arrow'>→</div><div class='diagram-box'><div class='diagram-title'>16-feature vector</div><div class='diagram-sub'>Identical ordered feature contract</div></div></div><div class='branch-grid'><div class='branch'><div class='branch-title'>Reference path — Python / Streamlit</div><div class='branch-sub'>AdaBoost inference → plots → 3D digital twin → interactive diagnosis. This is the analysis and visualization interface.</div></div><div class='branch'><div class='branch-title'>Deployment path — Embedded C / STM32</div><div class='branch-sub'>CMSIS-DSP FFT + C feature extraction + 50-tree AdaBoost → STM32F446RE firmware. This is the resource-constrained execution target.</div></div></div>""",unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True); st.markdown('### How the paths are unified'); st.markdown("""<div class='diagram'><div class='diagram-box'><div class='diagram-title'>Same input contract</div><div class='diagram-sub'>500 X + 500 Y samples</div></div><div class='diagram-arrow'>→</div><div class='diagram-box'><div class='diagram-title'>Parity verification</div><div class='diagram-sub'>Python ↔ C outputs checked on held-out test vectors</div></div><div class='diagram-arrow'>→</div><div class='diagram-box'><div class='diagram-title'>Equivalent diagnosis</div><div class='diagram-sub'>Same five-class fault decision</div></div></div>""",unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True); st.markdown('### End-to-end process'); st.markdown("""<div class='step-row'><div class='step-card'><div class='step-num'>01</div><div class='step-title'>Sense</div><div class='step-sub'>Capture vibration from the UAV frame</div></div><div class='step-card'><div class='step-num'>02</div><div class='step-title'>Window</div><div class='step-sub'>Collect 500 samples per axis</div></div><div class='step-card'><div class='step-num'>03</div><div class='step-title'>Extract</div><div class='step-sub'>Time + frequency features</div></div><div class='step-card'><div class='step-num'>04</div><div class='step-title'>Diagnose</div><div class='step-sub'>AdaBoost predicts one of five states</div></div><div class='step-card'><div class='step-num'>05</div><div class='step-title'>Deploy</div><div class='step-sub'>Equivalent C pipeline runs on STM32F446RE</div></div></div>""",unsafe_allow_html=True)
    st.info('The browser dashboard visualizes the Python reference path. The embedded path is validated independently through C parity, STM32 compilation, memory measurement and Renode execution. Physical NUCLEO-F446RE execution is the next hardware-validation step.')

st.caption('Scope: this prototype diagnoses five propeller-health classes using a controlled UAV vibration dataset. New compatible recordings can be evaluated through the same frozen inference pipeline, but uploads do not retrain the model. External deployment across different UAVs, RPM ranges, payloads, sensors and outdoor conditions requires additional labelled validation and, where necessary, retraining.')
