from common import *
FONTS="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&family=Work+Sans:wght@400;500;600&display=swap"
CSS="""
:root{--ink:#15161c;--indigo:#2d33b5;--paper:#fff;--mist:#eef0f7;--mute:#5d6070;--head:'Archivo',sans-serif;--body:'Work Sans',sans-serif}
.slide{background:var(--paper);color:var(--ink);font-family:var(--body)}
.h{font-family:var(--head);font-stretch:112%;font-weight:800;letter-spacing:-.01em;line-height:.98}
.cap{font:600 22px var(--body);letter-spacing:.14em;text-transform:uppercase}
.rule{position:absolute;height:3px;background:var(--ink)}
/* title */
.s1 .mark{position:absolute;left:110px;top:92px}
.s1 .mark b{font:800 34px var(--head);font-stretch:125%;letter-spacing:.02em;color:var(--indigo)}
.s1 .mark span{display:block;margin-top:10px}
.s1 .t{position:absolute;left:110px;top:230px;font-size:118px;width:980px}
.s1 .sub{position:absolute;left:110px;top:700px;width:820px;font:500 34px/1.35 var(--body)}
.s1 .photo{position:absolute;left:1000px;top:560px;width:500px;height:380px;background:var(--indigo);overflow:hidden}
.s1 .photo img{width:100%;height:100%;object-fit:cover;object-position:50% 50%;transform:scale(2.3);transform-origin:80% 56%;filter:grayscale(1) brightness(1.5) contrast(1.05)}
.s1 .photo:after{content:"";position:absolute;inset:0;background:var(--indigo);mix-blend-mode:multiply;opacity:.7}
.s1 .col{position:absolute;right:0;top:0;width:420px;height:1080px;background:var(--indigo);overflow:hidden}
.s1 .col span{position:absolute;left:50%;top:60px;transform:translateX(-50%);writing-mode:vertical-rl;font:800 168px/1 var(--head);font-stretch:112%;color:transparent;-webkit-text-stroke:3px #fff;letter-spacing:.02em}
.s1 .meta{position:absolute;left:0;bottom:90px;width:560px;height:130px;border-left:26px solid var(--indigo);padding:6px 0 0 54px;font:500 26px/1.5 var(--body)}
.s1 .meta b{font-weight:600;color:var(--indigo)}
/* results */
.s2 .left{position:absolute;left:0;top:0;width:620px;height:1080px;background:var(--indigo);color:#fff;padding:96px 70px}
.s2 .left .big{font:800 122px/.9 var(--head);font-stretch:112%;letter-spacing:-.03em;margin-top:44px}
.s2 .left p{font:500 27px/1.4 var(--body);margin-top:20px;max-width:440px}
.s2 .left hr{border:0;height:2px;background:#ffffff66;margin:54px 0 40px}
.s2 .left .mid{font:800 76px/1 var(--head);font-stretch:112%;letter-spacing:-.02em}
.s2 .right{position:absolute;left:720px;top:0;right:0;height:1080px;padding:96px 110px 0 0}
.s2 h2{font-size:60px;line-height:1.04;max-width:1000px}
.s2 .t{margin-top:44px;width:1090px;font-size:26px}
.s2 .t th{font:600 20px var(--body);letter-spacing:.14em;text-transform:uppercase;text-align:right;padding:0 0 14px;border-bottom:3px solid var(--ink);color:var(--ink)}
.s2 .t th:first-child,.s2 .t td:first-child{text-align:left}
.s2 .t td{padding:11px 0;border-bottom:1px solid #c9cbd6;text-align:right;color:#2b2d38}
.s2 .t td i{font-style:normal;color:#9a9db0;font-size:20px;margin-left:4px}
.s2 .t td.best{font-weight:700;color:var(--indigo)}
.s2 .t td.best i{color:var(--indigo)}
.s2 .note{margin-top:26px;font:500 24px/1.4 var(--body);color:var(--mute);max-width:1000px}
.pg{position:absolute;right:110px;bottom:46px;font:600 22px var(--body);color:var(--mute)}
.lab{position:absolute;right:70px;top:96px;writing-mode:vertical-rl;font:800 28px var(--head);font-stretch:112%;letter-spacing:.2em;color:var(--indigo)}
/* explainer */
.s3 h2{position:absolute;left:110px;top:110px;font-size:68px;width:1250px}
.s3 .cap.top{position:absolute;left:110px;top:60px;color:var(--indigo)}
.s3 .cols{position:absolute;left:110px;right:200px;top:400px;display:grid;grid-template-columns:repeat(4,1fr);gap:0}
.s3 .c{padding:0 30px 0 0;position:relative;margin-right:24px}
.s3 .c:nth-child(2){margin-top:60px}.s3 .c:nth-child(3){margin-top:120px}.s3 .c:nth-child(4){margin-top:180px}
.s3 .c .n{font:600 26px var(--body);color:var(--indigo);border-top:3px solid var(--ink);padding-top:16px;margin-bottom:26px;margin-right:0}
.s3 .c .pair{font:800 84px/1 var(--head);font-stretch:112%;letter-spacing:-.02em;white-space:nowrap}
.s3 .c .pair .arr{width:52px;height:21px;margin:0 8px}
.s3 .c h3{font:700 32px/1.2 var(--body);margin-top:24px}
.s3 .c p{font:400 26px/1.4 var(--body);color:var(--mute);margin-top:10px}
.s3 .c.hot{background:var(--indigo);color:#fff;padding:0 30px 44px 30px}
.s3 .c.hot .n{border-color:#fff;color:#fff;margin-top:0;padding-top:20px}
.s3 .c.hot p{color:#dfe1ff}
.s3 .foot{position:absolute;left:110px;bottom:78px;font:500 26px var(--body);color:var(--mute);border-top:3px solid var(--ink);padding-top:18px;width:1100px}
.s3 .rail{position:absolute;right:0;top:0;width:150px;height:1080px;background:var(--indigo)}
.s3 .rail span{position:absolute;left:50%;top:80px;transform:translateX(-50%);writing-mode:vertical-rl;font:800 100px var(--head);font-stretch:112%;color:transparent;-webkit-text-stroke:2px #fff;letter-spacing:.04em}
"""
def col(n,pair,title,desc,hot=False):
    a,b=pair
    return '<div class="c%s"><div class="n">%s</div><div class="pair">%s%s%s</div><h3>%s</h3><p>%s</p></div>' % (' hot' if hot else '',n,a,ARROW,b,title,desc)
s1='''<div class="s1"><div class="mark"><b>IS460</b><span class="cap">Final project presentation</span></div>
<div class="t h">Cleared<br>for Satisfaction</div>
<div class="sub">Evaluating synthetic data with tabular ML ensembles, transfer testing and explainable AI.</div>
<div class="photo"><img src="img/sunrise.jpg" alt="Wing of an airliner against a dusk horizon"></div>
<div class="col"><span>CLEARED</span></div>
<div class="meta"><b>Group 03</b><br>SMU · 2026</div></div>'''
s2='''<div class="s2"><div class="left"><div class="cap">Key result</div><div class="big">0.9954</div><p>TabM, trained and tested on real passengers (R→R AUC).</p><hr><div class="mid">0.9603</div><p>CatBoost, the best model on the synthetic copy (S→S AUC).</p></div>
<div class="right"><h2 class="h">TabM wins on real passengers. CatBoost wins on the synthetic copy.</h2>
%s<div class="note">AUC of ten models in four settings, rank in brackets. S = synthetic, R = real. Rows sorted by R→R.</div></div>
<div class="lab">RQ1 · MODEL LADDER</div><div class="pg">2</div></div>''' % table('t')
s3='''<div class="s3"><div class="cap top">Method · Validation</div><h2 class="h">How we test whether synthetic findings transfer</h2>
<div class="cols">%s%s%s%s</div>
<div class="foot">All models use fixed 5-fold stratified cross-validation (seed 42). Metric: ROC AUC.</div>
<div class="rail"><span>TEST</span></div></div>''' % (
 col('01',('S','S'),'Synthetic to synthetic','Kaggle-style validation.'),
 col('02',('S','R'),'Synthetic to real','Transfer to real passengers.',True),
 col('03',('R','S'),'Real to synthetic','Reverse transfer check.'),
 col('04',('R','R'),'Real to real','Reference performance.'))
open('d1.html','w').write(page('Direction 1 · Swiss Editorial / Indigo Technical',FONTS,CSS,[s1,s2,s3]))
print('d1 ok')
