from common import *
FONTS="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Figtree:wght@400;500;600&display=swap"
CSS="""
:root{--surf:#d5d6d9;--panel:#e7e7e9;--white:#f6f6f7;--steel:#9bb0c8;--ink:#0d0d0f;--mute:#6a6d75;--head:'Sora',sans-serif;--body:'Figtree',sans-serif}
.slide{background:var(--surf);color:var(--ink);font-family:var(--body)}
.pnl{position:absolute;background:var(--panel);border-radius:40px;box-shadow:0 2px 0 #fff9 inset,0 18px 40px -22px #0003}
.pnl.w{background:var(--white)}
.pnl.b{background:var(--steel)}
.logo{position:absolute;left:84px;top:70px;width:76px;height:76px;border-radius:50%;background:#000;color:#fff;font:700 24px/76px var(--head);text-align:center}
.eyebrow{font:500 24px var(--body);color:var(--mute)}
h1,h2,h3{font-family:var(--head);font-weight:700;letter-spacing:-.02em}
/* title */
.s1 h1{position:absolute;left:84px;top:190px;font-size:128px;line-height:.98;width:1000px}
.s1 .sub{position:absolute;right:84px;top:230px;width:620px;font:400 32px/1.4 var(--body);color:#2a2c33}
.s1 .hero{left:84px;top:520px;width:1752px;height:300px;overflow:hidden;border-radius:44px}
.s1 .hero img{width:100%;height:100%;object-fit:cover;object-position:50% 52%;filter:saturate(.8) contrast(1.05)}
.s1 .hero:after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,#d5d6d9aa,transparent 35%);}
.s1 .m1{left:84px;top:850px;width:420px;height:170px;padding:30px 36px;color:#0d1a2b}
.s1 .m2{left:524px;top:850px;width:1312px;height:170px;display:grid;grid-template-columns:repeat(3,1fr);padding:30px 20px}
.s1 .m2 div{padding:0 26px;border-left:2px solid #c5c6ca}.s1 .m2 div:first-child{border:0}
.s1 .m2 b{display:block;font:700 54px var(--head);letter-spacing:-.03em}
.s1 .m2 span{font:500 24px var(--body);color:var(--mute)}
.s1 .m1 b{font:700 44px var(--head)}.s1 .m1 span{display:block;font:500 24px var(--body);margin-top:8px;opacity:.8}
/* results */
.s2 h2{position:absolute;left:176px;top:78px;font-size:62px;line-height:1.05;width:1250px}
.s2 .logo{width:76px;height:76px}
.s2 .tab{left:84px;top:230px;width:1180px;height:770px;padding:34px 48px}
.s2 .t{width:100%;font-size:26px}
.s2 .t th{font:600 21px var(--body);color:var(--mute);text-align:right;padding:0 0 16px;border-bottom:2px solid #cfd0d4}
.s2 .t th:first-child,.s2 .t td:first-child{text-align:left}
.s2 .t td{padding:13px 0;border-bottom:1px solid #dcdde0;text-align:right}
.s2 .t td i{font-style:normal;color:#9a9ca3;font-size:20px;margin-left:4px}
.s2 .t td.best{font-weight:700}
.s2 .t td.best:not(:first-child){position:relative}
.s2 .t td.best b{}
.s2 .k1{left:1304px;top:230px;width:532px;height:330px;padding:40px 44px;color:#0d1a2b}
.s2 .k1 .big{font:700 110px/1 var(--head);letter-spacing:-.04em;margin-top:30px}
.s2 .k1 p{font:500 26px/1.35 var(--body);margin-top:12px}
.s2 .k2{left:1304px;top:588px;width:532px;height:200px;padding:34px 44px}
.s2 .k2 .big{font:700 64px/1 var(--head);letter-spacing:-.03em;margin-top:10px}
.s2 .k3{left:1304px;top:816px;width:532px;height:184px;padding:30px 44px}
.s2 .k3 p{font:500 26px/1.4 var(--body);margin-top:8px}
.lbl{font:600 21px var(--body);letter-spacing:.04em;color:var(--mute)}
.k1 .lbl{color:#31455f}
.s2 .hi{background:#fff;border-radius:12px}
/* explainer */
.s3 h2{position:absolute;left:176px;top:78px;font-size:60px;line-height:1.05;width:1400px}
.s3 .stage{left:84px;top:230px;width:1752px;height:150px;display:grid;grid-template-columns:repeat(4,1fr);padding:0 30px;align-items:center}
.s3 .st{display:flex;align-items:center;gap:20px;padding:0 20px}
.s3 .st i{font:700 26px/52px var(--head);font-style:normal;width:56px;height:56px;border-radius:50%;background:#e1e1e4;text-align:center;flex:none}
.s3 .st.on i{background:#000;color:#fff}
.s3 .st b{font:600 28px var(--head);display:block}.s3 .st span{font:400 22px/1.3 var(--body);color:var(--mute)}
.s3 .diag{left:84px;top:410px;width:1752px;height:420px}
.s3 .diag svg{position:absolute;inset:0;width:100%;height:100%}
.s3 .ds{position:absolute;top:110px;width:420px;height:200px;border-radius:32px;padding:34px 40px}
.s3 .ds b{font:700 44px var(--head);display:block}.s3 .ds span{font:500 26px/1.4 var(--body)}
.s3 .ds.s{left:90px;background:#f6f6f7}.s3 .ds.r{right:90px;background:var(--steel);color:#0d1a2b}
.s3 .tag{position:absolute;font:700 30px var(--head);padding:10px 22px;border-radius:999px;background:#e1e1e4}
.s3 .tag.hot{background:#000;color:#fff}
.s3 .cap{position:absolute;font:500 22px var(--body);color:var(--mute)}
.s3 .spec{top:860px;height:150px;width:566px;padding:28px 38px}
.s3 .spec b{font:700 40px var(--head);display:block;margin-top:4px}
"""
tags='''<div class="s3"><div class="logo">G3</div><h2>How we test whether synthetic findings transfer</h2>
<div class="pnl stage">
<div class="st"><i>1</i><div><b>Prepare</b><span>Shared features, fixed folds</span></div></div>
<div class="st"><i>2</i><div><b>Train</b><span>Ten models</span></div></div>
<div class="st on"><i>3</i><div><b>Evaluate</b><span>Four settings, ROC AUC</span></div></div>
<div class="st"><i>4</i><div><b>Interpret</b><span>Fidelity, SHAP, what-if</span></div></div></div>
<div class="pnl diag">
<svg viewBox="0 0 1752 420" fill="none" stroke="#0d0d0f" stroke-width="3">
<path d="M520 160 C 700 90, 1050 90, 1230 160" /><path d="M1222 150l16 14-20 6" stroke-linecap="square"/>
<path d="M1230 262 C 1050 335, 700 335, 520 262" /><path d="M528 252l-16 14 20 6" stroke-linecap="square"/>
<path d="M200 110 C 120 40, 360 40, 300 110" stroke-width="2.4"/><path d="M1452 110 C 1392 40, 1632 40, 1552 110" stroke-width="2.4"/>
</svg>
<div class="ds s"><span class="lbl">Synthetic survey</span><b>699,635</b><span>Kaggle Playground S6E10</span></div>
<div class="ds r"><span class="lbl" style="color:#31455f">Real survey</span><b>129,880</b><span>Original survey dataset</span></div>
<div class="tag hot" style="left:800px;top:10px">S→R</div><div class="cap" style="left:690px;top:62px;width:380px;text-align:center;left:686px">Transfer to real passengers</div>
<div class="tag" style="left:760px;top:350px">R→S</div><div class="cap" style="left:900px;top:362px;width:280px">Reverse check</div>
<div class="tag" style="left:84px;top:22px">S→S</div><div class="cap" style="left:26px;top:318px;width:380px;text-align:left;left:110px">Kaggle-style validation</div>
<div class="tag" style="left:1480px;top:22px">R→R</div><div class="cap" style="left:1448px;top:318px;width:300px;text-align:right;right:110px;left:auto">Reference performance</div></div>
<div class="pnl w spec" style="left:84px"><span class="lbl">Cross-validation</span><b>5-fold stratified</b></div>
<div class="pnl spec" style="left:677px"><span class="lbl">Random seed</span><b>42</b></div>
<div class="pnl b spec" style="left:1270px;color:#0d1a2b"><span class="lbl" style="color:#31455f">Metric</span><b>ROC AUC</b></div></div>'''
s1='''<div class="s1"><div class="logo">G3</div><div class="eyebrow" style="position:absolute;left:184px;top:92px">IS460 · Final project presentation</div>
<h1>Cleared for Satisfaction</h1><div class="sub">Evaluating synthetic data with tabular ML ensembles, transfer testing and explainable AI.</div>
<div class="pnl hero"><img src="img/sunrise.jpg" alt="Wing of an airliner against a dusk horizon"></div>
<div class="pnl b m1"><span class="lbl" style="color:#31455f">Group 03</span><b>SMU · 2026</b><span>Final report</span></div>
<div class="pnl w m2"><div><b>699,635</b><span>synthetic passengers</span></div><div><b>129,880</b><span>real passengers</span></div><div><b>10 × 4</b><span>models × train/test settings</span></div></div></div>'''
s2='''<div class="s2"><div class="logo">G3</div><h2>TabM wins on real passengers. CatBoost wins on the synthetic copy.</h2>
<div class="pnl w tab">%s</div>
<div class="pnl b k1"><span class="lbl">Best on real · R→R AUC</span><div class="big">0.9954</div><p>TabM, trained and tested on real passengers.</p></div>
<div class="pnl k2"><span class="lbl">Best on synthetic · S→S AUC</span><div class="big">0.9603</div></div>
<div class="pnl w k3"><span class="lbl">Rank in brackets</span><p>S = synthetic, R = real. Sorted by R→R.</p></div></div>''' % table('t')
open('d2.html','w').write(page('Direction 2 · Soft Industrial / Research Console',FONTS,CSS,[s1,s2,tags]))
print('d2 ok')
