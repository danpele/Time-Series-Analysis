#!/bin/zsh
# usage: cd <TSA repo>/web && ../latex/check_web_page.sh ch4_ro.html [width]      (default width 1440; 390 for a phone)
# needs a server on localhost:$PORT (default 8791) serving the TSA repo root: cd <TSA repo> && python3 -m http.server 8791
# (8765 is the MFM server; another port: PORT=8800 ../latex/check_web_page.sh ...).
# The page is loaded in an iframe of the given width (headless Chrome has no window narrower than 500 px).
# Prints: JS errors, whether MathJax finished, MathJax errors (mjx-merror), frames, fallback frames, page width / frame width,
# and the elements wider than the frame (wide=..., at most 5: tag.class).
SP=${TMPDIR:-/tmp}; PORT=${PORT:-8791}
f=$1; w=${2:-1440}; t=_check_$RANDOM.html; wr=_wrap_$RANDOM.html
python3 -c "
s=open('$f',encoding='utf-8').read()
s=s.replace('<head>','<head><script>window.__err=[];addEventListener(\"error\",e=>window.__err.push(e.message));</script>',1)
open('$t','w',encoding='utf-8').write(s)
js='''setTimeout(()=>{const f=document.querySelector(\"iframe\"),W=f.contentWindow,D=f.contentDocument,iw=W.innerWidth;
const wide=[...D.querySelectorAll(\"main *, header *, nav *\")].filter(e=>{const r=e.getBoundingClientRect();return r.right>iw+1&&!e.closest(\".dm, .tbl, pre, mjx-container\")}).slice(0,5).map(e=>e.tagName.toLowerCase()+\".\"+String(e.className).replace(/ /g,\".\"));
document.body.setAttribute(\"data-err\",JSON.stringify(W.__err)+\" mj=\"+(D.body.getAttribute(\"data-mj\")||\"no\")+\" merror=\"+D.querySelectorAll(\"mjx-merror\").length+\" frames=\"+D.querySelectorAll(\"article.frame\").length+\" fallback=\"+D.querySelectorAll(\"article.fb\").length+\" w=\"+D.documentElement.scrollWidth+\"/\"+iw+\" wide=\"+wide.join(\",\"))},25000)'''
open('$wr','w',encoding='utf-8').write('<!DOCTYPE html><html><body style=\"margin:0\"><iframe src=\"$t\" style=\"width:${w}px;height:900px;border:0\"></iframe><script>'+js+'</script></body></html>')"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars --user-data-dir=$SP/chrome-prof-$RANDOM --window-size=$(( w > 500 ? w : 500 )),900 --virtual-time-budget=35000 --dump-dom "http://localhost:$PORT/web/$wr" 2>/dev/null | grep -o 'data-err="[^"]*"'
python3 -c "import os; os.remove('$t'); os.remove('$wr')"
