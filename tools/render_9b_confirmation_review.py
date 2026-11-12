"""Render a local blind-review form using only public queue fields."""
import argparse
import json
from pathlib import Path

from experiment_common import read_rows


def render(queue):
    rows = read_rows(queue)
    fields = {'review_id', 'image', 'width', 'height', 'referring_expression'}
    if not rows or any(set(row) != fields for row in rows):
        raise ValueError('Unexpected fields in blind review queue')
    data = json.dumps(rows, ensure_ascii=False).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>9B 定位人工盲审</title>
<style>
body{font:16px system-ui,sans-serif;background:#f4f6fa;color:#182032;max-width:900px;margin:2rem auto;padding:0 1rem}
button,input,select{font:inherit;padding:.35rem;margin:.25rem}
canvas{display:block;max-width:100%;max-height:65vh;touch-action:none;background:#e7eaf0}
</style>
<h1>9B 定位人工盲审</h1>
<p>独立阅读表达，在原图拖框圈出唯一指代对象。指代不唯一、不可见或图像有问题时标记并说明。请勿查原数据或与另一位审阅者交流。</p>
<label>审阅者代号 <input id="reviewer" autocomplete="off"></label>
<label>恢复自己的 JSON <input id="import" type="file" accept=".json"></label>
<button id="export">导出审阅 JSON</button>
<div><button id="prev">上一题</button><output id="progress"></output><button id="next">下一题</button></div>
<h2 id="expression"></h2><canvas id="image"></canvas>
<div><label>判断 <select id="verdict"><option value="">请选择</option><option value="accept">唯一且可框出</option><option value="ambiguous">歧义</option><option value="unusable">不可用</option></select></label>
<label>备注 <input id="note" autocomplete="off"></label><button id="save">保存本题</button></div>
<p id="message" aria-live="polite"></p>
<script id="data" type="application/json">__DATA__</script>
<script>
const el = id => document.getElementById(id), rows = JSON.parse(el("data").textContent);
const canvas = el("image"), ctx = canvas.getContext("2d"), image = new Image();
let index=0, start=null, box=null, answers={}, loaded=false;
function draw(){
  if(!loaded) return;
  ctx.drawImage(image,0,0,canvas.width,canvas.height);
  if(box){ctx.strokeStyle="#00e5ff";ctx.lineWidth=3;ctx.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);}
}
function show(){
  const row=rows[index], answer=answers[row.review_id];
  el("progress").textContent=(index+1)+"/"+rows.length+" · 已保存 "+Object.keys(answers).length;
  el("expression").textContent=row.referring_expression;
  el("verdict").value=answer?.verdict ?? ""; el("note").value=answer?.note ?? "";
  box=answer?.box ?? null; loaded=false; start=null;
  canvas.width=row.width; canvas.height=row.height;
  image.src=row.image;
}
image.onload=()=>{loaded=true;draw();};
image.onerror=()=>{loaded=false;el("message").textContent="图像缺失，无法保存本题";};
function point(event){
  const rect=canvas.getBoundingClientRect();
  return [[event.clientX-rect.left,rect.width,canvas.width],[event.clientY-rect.top,rect.height,canvas.height]]
    .map(([v,extent,size])=>Math.max(0,Math.min(size,Math.round(v/extent*size))));
}
canvas.onpointerdown=event=>{if(loaded){start=point(event);canvas.setPointerCapture(event.pointerId);}};
canvas.onpointermove=event=>{if(start){const end=point(event);box=[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])];draw();}};
canvas.onpointerup=()=>{start=null;};
el("prev").onclick=()=>{if(index>0){index--;show();}};
el("next").onclick=()=>{if(index+1<rows.length){index++;show();}};
el("save").onclick=()=>{
  const verdict=el("verdict").value, note=el("note").value.trim();
  if(!loaded || !verdict || (verdict==="accept" ? (!box || box[0]>=box[2] || box[1]>=box[3]) : !note)){
    el("message").textContent="请完成判断。可用题需拖框，其余需写明原因。";return;
  }
  answers[rows[index].review_id]={verdict,note,box:verdict==="accept"?box:null};
  el("message").textContent="已保存本题，请定期导出 JSON。";
  if(index+1<rows.length) index++;
  show();
};
el("export").onclick=()=>{
  const reviewer=el("reviewer").value.trim();
  if(!reviewer){el("message").textContent="请输入自己的审阅者代号";return;}
  const blob=new Blob([JSON.stringify({reviewer,answers},null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob), link=document.createElement("a");
  link.href=url;link.download="blind-review.json";link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
};
el("import").onchange=async event=>{
  try{
    const record=JSON.parse(await event.target.files[0].text());
    const valid=new Set(rows.map(row=>row.review_id));
    if(!record.reviewer || !record.answers || !Object.keys(record.answers).every(id=>valid.has(id))) throw Error();
    answers=record.answers;el("reviewer").value=record.reviewer;show();
  }catch{el("message").textContent="无法载入这份审阅文件";}
};
show();
</script></html>'''
    queue.with_name('review.html').write_text(page.replace('__DATA__', data), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queue', type=Path)
    render(parser.parse_args().queue)
