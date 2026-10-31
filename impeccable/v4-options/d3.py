from common import *
FONTS="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&family=Onest:wght@400;500;600&display=swap"
CSS="""
:root{--sky:#b3c3d9;--sky2:#d8e2ef;--ink:#0a0a0c;--mute:#5a6475;--head:'Bricolage Grotesque',sans-serif;--body:'Onest',sans-serif}
.slide{background:linear-gradient(180deg,#a9bad2,#c3d0e2);color:var(--ink);font-family:var(--body)}
.card{position:absolute;background:#fff;border-radius:48px;box-shadow:0 24px 50px -30px #1b2c4a66}
.grad{background:linear-gradient(180deg,#fff 0%,#d5e0ee 100%)}
.pill{display:inline-block;background:#000;color:#fff;font:500 24px var(--body);padding:10px 26px;border-radius:999px}
h1,h2{font-family:var(--head);font-weight:800;letter-spacing:-.035em;line-height:.95}
/* title */
.s1 .card{left:70px;top:60px;width:1780px;height:960px}
.s1 h1{position:absolute;left:96px;top:96px;font-size:158px;width:1100px;line-height:.98}
.s1 .sub{position:absolute;right:96px;top:120px;width:480px;text-align:right;font:400 32px/1.4 var(--body)}
.s1 .pill{position:absolute;right:96px;top:300px}
.s1 .ph{position:absolute;left:96px;right:96px;top:450px;height:430px;border-radius:40px;overflow:hidden;background:#0d1a30}
.s1 .ph img{width:100%;height:100%;object-fit:cover;object-position:50% 55%}
.s1 .foot{position:absolute;left:96px;bottom:50px;font:500 26px var(--body);color:var(--mute)}
/* results */
.s2 .left{position:absolute;left:70px;top:60px;width:700px;height:960px;border-radius:48px 0 0 48px;padding:84px 70px;background:linear-gradient(180deg,#f7fafd 0%,#cdd9e8 100%)}
.s2 .left h2{font-size:92px}
.s2 .left .m{margin-top:70px;padding-top:26px;border-top:3px solid #000}
.s2 .left .m b{font:800 120px/1 var(--head);letter-spacing:-.04em;display:block}
.s2 .left .m span{font:500 28px/1.35 var(--body);color:#26334a;display:block;margin-top:8px}
.s2 .left .m2 b{font-size:84px}
.s2 .right{position:absolute;left:770px;top:60px;width:1080px;height:960px;background:#fff;border-radius:0 48px 48px 0;padding:76px 70px}
.s2 .t{width:100%;font-size:26px}
.s2 .t th{font:600 22px var(--body);text-align:right;padding:0 0 16px;color:var(--mute);border-bottom:2px solid #000}
.s2 .t th:first-child,.s2 .t td:first-child{text-align:left}
.s2 .t td{padding:14px 0;border-bottom:1px solid #e3e8f0;text-align:right;white-space:nowrap}.s2 .t td:first-child{white-space:normal}
.s2 .t td i{font-style:normal;color:#9aa6b8;font-size:20px;margin-left:4px}
.s2 .t td.best{font-weight:700}
.s2 .t td.best:not(:first-child){background:#d8e4f3;border-radius:0}
.s2 .cap{font:500 24px/1.4 var(--body);color:var(--mute);margin-top:24px}
.s2 .tag{margin-bottom:34px}
/* explainer */
.s3 .card{left:70px;top:60px;width:1780px;height:960px}
.s3 h2{position:absolute;left:96px;top:96px;font-size:104px;width:1450px}
.s3 .pill{position:absolute;right:96px;top:100px}
.s3 .mx{position:absolute;left:96px;top:380px;width:1588px;height:420px}
.s3 .axis{position:absolute;font:600 24px var(--body);color:var(--mute)}
.s3 .cell{position:absolute;width:640px;height:190px;border-radius:32px;padding:30px 40px;background:linear-gradient(180deg,#f4f8fc,#d9e4f1)}
.s3 .cell b{font:800 66px/1 var(--head);letter-spacing:-.03em;display:flex;align-items:center;gap:6px}
.s3 .cell b .arr{width:56px;height:23px}
.s3 .cell span{font:500 26px/1.35 var(--body);color:#26334a;display:block;margin-top:10px}
.s3 .cell.hot{background:#000;color:#fff}.s3 .cell.hot span{color:#cfd8e6}
.s3 .row3{position:absolute;left:96px;right:96px;bottom:110px;display:grid;grid-template-columns:repeat(3,1fr);gap:0;border-top:3px solid #000;padding-top:22px}
.s3 .row3 div{padding-right:40px}.s3 .row3 b{font:800 40px var(--head);display:block}.s3 .row3 span{font:400 26px var(--body);color:var(--mute)}
"""
def cell(x,y,a,b,t,d,hot=False):
    return '<div class="cell%s" style="left:%spx;top:%spx"><b>%s%s%s</b><span><strong style="font-weight:600">%s.</strong> %s</span></div>'%(' hot' if hot else '',x,y,a,ARROW,b,t,d)
s1='''<div class="s1"><div class="card"></div><h1>Cleared for Satisfaction</h1>
<div class="sub">Evaluating synthetic data with tabular ML ensembles, transfer testing and explainable AI.</div>
<span class="pill">IS460 · Final project presentation</span>
<div class="ph"><img src="img/sunrise.jpg" alt="Wing of an airliner against a dusk horizon"></div>
<div class="foot">Group 03 · SMU · 2026</div></div>'''
s2='''<div class="s2"><div class="left"><span class="pill tag">Results · RQ1</span><h2>TabM leads on real passengers</h2>
<div class="m"><b>0.9954</b><span>TabM, trained and tested on real passengers (R→R AUC)</span></div>
<div class="m m2"><b>0.9603</b><span>CatBoost, best on the synthetic copy (S→S AUC)</span></div></div>
<div class="right">%s<div class="cap">AUC of ten models in four settings, rank in brackets. S = synthetic, R = real. Sorted by R→R.</div></div></div>''' % table('t')
s3='''<div class="s3"><div class="card"></div><h2>How we test whether synthetic findings transfer</h2><span class="pill">Method · Validation</span>
<div class="mx">
<div class="axis" style="left:320px;top:0">Tested on synthetic</div><div class="axis" style="left:1000px;top:0">Tested on real</div>
<div class="axis" style="left:0;top:96px;width:200px">Trained on<br>synthetic</div><div class="axis" style="left:0;top:306px;width:200px">Trained on<br>real</div>
%s%s%s%s</div>
<div class="row3"><div><b>5-fold</b><span>stratified cross-validation</span></div><div><b>Seed 42</b><span>one fixed split per dataset</span></div><div><b>ROC AUC</b><span>performance metric</span></div></div></div>''' % (
 cell(220,40,'S','S','Synthetic to synthetic','Kaggle-style validation'),
 cell(900,40,'S','R','Synthetic to real','Transfer to real passengers',True),
 cell(220,250,'R','S','Real to synthetic','Reverse transfer check'),
 cell(900,250,'R','R','Real to real','Reference performance'))
open('d3.html','w').write(page('Direction 3 · Airy Modular / Sky-Blue Narrative',FONTS,CSS,[s1,s2,s3]))
print('d3 ok')
