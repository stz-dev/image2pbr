"""Interactive real-time 3D viewer (WebGL2, no external libraries, works offline).

build_document() returns one self-contained HTML page with the maps embedded as images.
It is shown inside the app through an <iframe srcdoc> and can also be saved as a standalone file.
All sliders (strength of every map, light, tiling, ...) run in the browser, so they react instantly.
"""
from __future__ import annotations

import base64
import html as _html
import json

import cv2
import numpy as np

from . import export

LAYERS = ("basecolor", "roughness", "metallic", "normal", "height", "ao")


def _data_uri(arr: np.ndarray, size: int, jpeg: bool = False) -> str:
    a = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    if jpeg:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        mime, data = "image/jpeg", buf.tobytes()
    else:
        mime, data = "image/png", export.png_bytes(a)
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def build_document(albedo: np.ndarray, maps: dict, size: int = 512) -> str:
    """albedo: RGB uint8. maps: name -> uint8 array (roughness, metallic, normal, height, ao)."""
    uris = {"basecolor": _data_uri(albedo, size, jpeg=True)}
    for name in LAYERS[1:]:
        arr = maps.get(name)
        if arr is None:
            continue
        if arr.dtype == np.uint16:
            arr = (arr >> 8).astype(np.uint8)
        uris[name] = _data_uri(arr, size)
    cfg = {"maps": uris, "size": size}
    return DOC_TEMPLATE.replace("__CONFIG__", json.dumps(cfg))


def iframe(doc: str, height: int = 700) -> str:
    return (f'<iframe srcdoc="{_html.escape(doc, quote=True)}" '
            f'style="width:100%;height:{height}px;border:0;border-radius:10px;background:#1c1c1f" '
            f'allow="fullscreen" title="Image2PBR 3D viewer"></iframe>')


DOC_TEMPLATE = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Image2PBR 3D viewer</title>
<style>
html,body{margin:0;height:100%;background:#1c1c1f;color:#d8d8dc;font:12px/1.35 system-ui,sans-serif;overflow:hidden}
#wrap{display:flex;height:100%}
#stage{position:relative;flex:1;min-width:0}
#cv{width:100%;height:100%;display:block;touch-action:none;cursor:grab}
#cv.drag{cursor:grabbing}
#note{position:absolute;left:10px;bottom:8px;right:10px;color:#9a9aa3;pointer-events:none;font-size:11px}
#panel{width:262px;overflow-y:auto;padding:8px 10px;box-sizing:border-box;background:#26262b;border-left:1px solid #111}
h4{margin:12px 0 4px;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#8d8d97}
h4:first-child{margin-top:2px}
.row{margin:5px 0}
.row .top{display:flex;justify-content:space-between}
.row.off{opacity:.38}
.row .val{color:#9ec1ff;font-variant-numeric:tabular-nums}
input[type=range]{width:100%;margin:2px 0;accent-color:#6ea8ff}
select,button{background:#35353c;color:#e6e6ea;border:1px solid #4a4a53;border-radius:6px;padding:4px 8px;font:inherit}
button{cursor:pointer}button:hover{background:#41414a}
.btns{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
label.chk{display:flex;align-items:center;gap:6px;margin:6px 0}
</style></head>
<body>
<div id="wrap">
  <div id="stage"><canvas id="cv"></canvas><div id="note">Loading...</div></div>
  <div id="panel"></div>
</div>
<script>
(function () {
'use strict';
var CFG = __CONFIG__;
var canvas = document.getElementById('cv');
var note = document.getElementById('note');
var panel = document.getElementById('panel');
var gl = canvas.getContext('webgl2', {antialias: true, alpha: false, preserveDrawingBuffer: true});
if (!gl) { note.textContent = 'WebGL2 is not available in this browser/GPU.'; return; }

function has(name) { return !!CFG.maps[name]; }

// ---------------------------------------------------------------- controls
var CONTROLS = [
  {group: 'Maps - strength (0 = off, 1 = as generated)'},
  {id: 'kBase',   label: 'Base colour',         min: 0, max: 1,   step: 0.01,  def: 1,   map: 'basecolor'},
  {id: 'kRough',  label: 'Roughness',           min: 0, max: 2,   step: 0.01,  def: 1,   map: 'roughness'},
  {id: 'kMetal',  label: 'Metallic',            min: 0, max: 2,   step: 0.01,  def: 1,   map: 'metallic'},
  {id: 'kNormal', label: 'Normal',              min: 0, max: 3,   step: 0.01,  def: 1,   map: 'normal'},
  {id: 'kAO',     label: 'Ambient occlusion',   min: 0, max: 2,   step: 0.01,  def: 1,   map: 'ao'},
  {id: 'kBump',   label: 'Height as bump',      min: 0, max: 4,   step: 0.01,  def: has('normal') ? 0 : 1, map: 'height'},
  {id: 'kDisp',   label: 'Height as displacement', min: 0, max: 0.3, step: 0.005, def: 0.015, map: 'height'},
  {group: 'Light and surface'},
  {id: 'lightAz', label: 'Light direction',     min: -180, max: 180, step: 1,   def: 40},
  {id: 'lightEl', label: 'Light height',        min: 5,  max: 89,  step: 1,    def: 45},
  {id: 'lightI',  label: 'Light intensity',     min: 0,  max: 10,  step: 0.1,  def: 3.5},
  {id: 'envI',    label: 'Environment light',   min: 0,  max: 2,   step: 0.01, def: 1},
  {id: 'tile',    label: 'Tiling',              min: 1,  max: 8,   step: 0.5,  def: 2}
];
var STORE_KEY = 'image2pbr_viewer_v2';
var values = {}, shape = 'sphere', auto = false;
try {
  var saved = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
  values = saved.values || {}; shape = saved.shape || shape; auto = !!saved.auto;
} catch (e) {}
function persist() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify({values: values, shape: shape, auto: auto})); } catch (e) {}
}
var dirty = true;
function requestRender() { dirty = true; }

var widgets = {};
function buildPanel() {
  var h = '<h4>Shape</h4><div class="row"><select id="shape">' +
    '<option value="sphere">Sphere</option><option value="plane">Plane</option><option value="cube">Cube</option>' +
    '</select></div>';
  CONTROLS.forEach(function (c) {
    if (c.group) { h += '<h4>' + c.group + '</h4>'; return; }
    var missing = c.map && !has(c.map);
    var v = (values[c.id] !== undefined) ? values[c.id] : c.def;
    values[c.id] = v;
    h += '<div class="row' + (missing ? ' off' : '') + '" id="row_' + c.id + '"><div class="top"><span>' + c.label +
      (missing ? ' (not generated)' : '') + '</span><span class="val" id="val_' + c.id + '"></span></div>' +
      '<input type="range" id="in_' + c.id + '" min="' + c.min + '" max="' + c.max + '" step="' + c.step +
      '" value="' + v + '"' + (missing ? ' disabled' : '') + '></div>';
  });
  h += '<label class="chk"><input type="checkbox" id="auto"> Auto-rotate</label>' +
    '<div class="btns"><button id="resetView">Reset view</button><button id="resetAll">Reset all</button>' +
    '<button id="shot">Save PNG</button></div>' +
    '<h4>Controls</h4><div style="color:#8d8d97">Drag: rotate &middot; Wheel: zoom &middot; Double-click: reset view</div>';
  panel.innerHTML = h;
  CONTROLS.forEach(function (c) {
    if (c.group) return;
    var el = document.getElementById('in_' + c.id), out = document.getElementById('val_' + c.id);
    widgets[c.id] = {el: el, out: out, c: c};
    function show() { out.textContent = Number(el.value).toFixed(c.step < 0.05 ? 2 : (c.step < 1 ? 1 : 0)); }
    show();
    el.addEventListener('input', function () { values[c.id] = parseFloat(el.value); show(); persist(); requestRender(); });
  });
  var sh = document.getElementById('shape'); sh.value = shape;
  sh.addEventListener('change', function () { shape = sh.value; persist(); setMesh(); resetView(); });
  var au = document.getElementById('auto'); au.checked = auto;
  au.addEventListener('change', function () { auto = au.checked; persist(); requestRender(); });
  document.getElementById('resetView').addEventListener('click', resetView);
  document.getElementById('resetAll').addEventListener('click', function () {
    CONTROLS.forEach(function (c) {
      if (c.group) return;
      values[c.id] = c.def; widgets[c.id].el.value = c.def;
      widgets[c.id].out.textContent = Number(c.def).toFixed(c.step < 0.05 ? 2 : (c.step < 1 ? 1 : 0));
    });
    persist(); resetView();
  });
  document.getElementById('shot').addEventListener('click', function () {
    draw(0);
    canvas.toBlob(function (b) {
      var a = document.createElement('a'); a.href = URL.createObjectURL(b);
      a.download = 'image2pbr_preview.png'; document.body.appendChild(a); a.click(); a.remove();
    });
  });
}

// ---------------------------------------------------------------- math
function perspective(fovy, aspect, near, far) {
  var f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
function mul(a, b) {
  var o = new Float32Array(16);
  for (var c = 0; c < 4; c++) for (var r = 0; r < 4; r++) {
    var s = 0; for (var k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}
function rotX(a) { var c = Math.cos(a), s = Math.sin(a); return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]); }
function rotY(a) { var c = Math.cos(a), s = Math.sin(a); return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]); }
function translate(x, y, z) { return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]); }

// ---------------------------------------------------------------- geometry
function MeshBuilder() { this.pos = []; this.nor = []; this.uv = []; this.idx = []; this.n = 0; }
MeshBuilder.prototype.grid = function (nx, ny, fn) {
  var base = this.n;
  for (var j = 0; j <= ny; j++) for (var i = 0; i <= nx; i++) {
    var r = fn(i / nx, j / ny);
    this.pos.push(r.p[0], r.p[1], r.p[2]); this.nor.push(r.n[0], r.n[1], r.n[2]); this.uv.push(r.uv[0], r.uv[1]);
  }
  for (var j2 = 0; j2 < ny; j2++) for (var i2 = 0; i2 < nx; i2++) {
    var a = base + j2 * (nx + 1) + i2, b = a + nx + 1;
    this.idx.push(a, b, a + 1, b, b + 1, a + 1);
  }
  this.n += (nx + 1) * (ny + 1);
};
function buildMesh(kind) {
  var m = new MeshBuilder();
  if (kind === 'sphere') {
    m.grid(192, 96, function (u, v) {
      var phi = u * 2 * Math.PI, th = v * Math.PI, s = Math.sin(th);
      var p = [s * Math.sin(phi), Math.cos(th), s * Math.cos(phi)];
      return {p: p, n: p, uv: [u * 2, 1 - v]};
    });
  } else if (kind === 'plane') {
    m.grid(192, 192, function (u, v) { return {p: [(u * 2 - 1) * 1.25, (v * 2 - 1) * 1.25, 0], n: [0, 0, 1], uv: [u, v]}; });
  } else {
    var h = 0.8, faces = [
      [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
      [[1, 0, 0], [0, 0, -1], [0, 1, 0]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
      [[0, 1, 0], [1, 0, 0], [0, 0, -1]], [[0, -1, 0], [1, 0, 0], [0, 0, 1]]];
    faces.forEach(function (f) {
      m.grid(64, 64, function (u, v) {
        var p = [0, 0, 0];
        for (var i = 0; i < 3; i++) p[i] = h * (f[0][i] + f[1][i] * (2 * u - 1) + f[2][i] * (2 * v - 1));
        return {p: p, n: f[0], uv: [u, v]};
      });
    });
  }
  return m;
}

// ---------------------------------------------------------------- shaders
var VS = '#version 300 es\n' +
'in vec3 aPos; in vec3 aNor; in vec2 aUv;\n' +
'uniform mat4 uProj, uView, uModel; uniform float uTile, uDisp; uniform sampler2D tHeight;\n' +
'out vec3 vPos; out vec3 vNor; out vec2 vUv;\n' +
'void main(){\n' +
'  vec2 uv = aUv * uTile;\n' +
'  float h = textureLod(tHeight, uv, 0.0).r - 0.5;\n' +
'  vec3 p = aPos + aNor * h * uDisp;\n' +
'  mat4 mv = uView * uModel;\n' +
'  vec4 vp = mv * vec4(p, 1.0);\n' +
'  vPos = vp.xyz; vNor = mat3(mv) * aNor; vUv = uv;\n' +
'  gl_Position = uProj * vp;\n' +
'}\n';

var FS = '#version 300 es\n' +
'precision highp float;\n' +
'in vec3 vPos; in vec3 vNor; in vec2 vUv;\n' +
'uniform sampler2D tBase, tRough, tMetal, tNormal, tHeight, tAO;\n' +
'uniform float kBase, kRough, kMetal, kNormal, kAO, kBump, uLightI, uEnvI;\n' +
'uniform vec3 uLight; uniform vec2 uTexel;\n' +
'out vec4 fragColor;\n' +
'const float PI = 3.14159265359;\n' +
'mat3 cotangentFrame(vec3 N, vec3 p, vec2 uv){\n' +
'  vec3 dp1 = dFdx(p), dp2 = dFdy(p); vec2 duv1 = dFdx(uv), duv2 = dFdy(uv);\n' +
'  vec3 dp2perp = cross(dp2, N), dp1perp = cross(N, dp1);\n' +
'  vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;\n' +
'  vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;\n' +
'  float invmax = inversesqrt(max(max(dot(T, T), dot(B, B)), 1e-12));\n' +
'  return mat3(T * invmax, B * invmax, N);\n' +
'}\n' +
'vec3 skyColor(vec3 d, float rough){\n' +
'  float w = 0.12 + rough * 0.6;\n' +
'  float t = smoothstep(-w, w, d.y);\n' +
'  return mix(vec3(0.14, 0.11, 0.09), vec3(0.55, 0.68, 0.90), t);\n' +
'}\n' +
'void main(){\n' +
'  vec3 N = normalize(vNor);\n' +
'  vec3 V = normalize(-vPos);\n' +
'  if (dot(N, V) < 0.0) N = -N;\n' +
'  vec3 albedo = pow(mix(vec3(0.5), texture(tBase, vUv).rgb, kBase), vec3(2.2));\n' +
'  float rough = clamp(mix(0.5, texture(tRough, vUv).r, kRough), 0.04, 1.0);\n' +
'  float metal = clamp(texture(tMetal, vUv).r * kMetal, 0.0, 1.0);\n' +
'  float ao = clamp(mix(1.0, texture(tAO, vUv).r, kAO), 0.0, 1.0);\n' +
'  vec3 nt = texture(tNormal, vUv).xyz * 2.0 - 1.0;\n' +
'  vec2 sl = nt.xy / max(nt.z, 0.05) * kNormal;\n' +
'  float hl = texture(tHeight, vUv - vec2(uTexel.x, 0.0)).r, hr = texture(tHeight, vUv + vec2(uTexel.x, 0.0)).r;\n' +
'  float hd = texture(tHeight, vUv - vec2(0.0, uTexel.y)).r, hu = texture(tHeight, vUv + vec2(0.0, uTexel.y)).r;\n' +
'  sl += vec2(hl - hr, hd - hu) * kBump * 2.0 / (uTexel.x * 1024.0);\n' +
'  vec3 n = normalize(cotangentFrame(N, vPos, vUv) * vec3(sl, 1.0));\n' +
'  vec3 L = normalize(uLight), H = normalize(L + V);\n' +
'  float NoL = max(dot(n, L), 0.0), NoV = max(dot(n, V), 1e-4), NoH = max(dot(n, H), 0.0), VoH = max(dot(V, H), 0.0);\n' +
'  float a = max(rough * rough, 0.002), a2 = a * a;\n' +
'  float d = NoH * NoH * (a2 - 1.0) + 1.0;\n' +
'  float D = a2 / (PI * d * d);\n' +
'  float Vis = 0.5 / (NoL * sqrt(NoV * NoV * (1.0 - a2) + a2) + NoV * sqrt(NoL * NoL * (1.0 - a2) + a2) + 1e-4);\n' +
'  vec3 F0 = mix(vec3(0.04), albedo, metal);\n' +
'  vec3 F = F0 + (1.0 - F0) * pow(1.0 - VoH, 5.0);\n' +
'  vec3 diffuse = (1.0 - metal) * albedo / PI;\n' +
'  vec3 direct = (diffuse * (1.0 - F) + D * Vis * F) * NoL * uLightI * mix(1.0, ao, 0.6);\n' +
'  vec3 irr = mix(vec3(0.14, 0.11, 0.09), vec3(0.55, 0.68, 0.90), n.y * 0.5 + 0.5);\n' +
'  vec3 Fe = F0 + (max(vec3(1.0 - rough), F0) - F0) * pow(1.0 - NoV, 5.0);\n' +
'  vec3 amb = (1.0 - metal) * albedo * irr * uEnvI * ao\n' +
'           + skyColor(reflect(-V, n), rough) * Fe * uEnvI * ao * (1.0 - 0.5 * rough);\n' +
'  vec3 c = direct + amb;\n' +
'  c = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14);\n' +
'  fragColor = vec4(pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);\n' +
'}\n';

function compile(type, src) {
  var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('Shader error: ' + gl.getShaderInfoLog(s));
  return s;
}
var prog = gl.createProgram();
try {
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('Link error: ' + gl.getProgramInfoLog(prog));
} catch (e) { note.textContent = String(e.message || e); console.error(e); return; }
var U = {};
['uProj', 'uView', 'uModel', 'uTile', 'uDisp', 'kBase', 'kRough', 'kMetal', 'kNormal', 'kAO', 'kBump', 'uLightI',
 'uEnvI', 'uLight', 'uTexel', 'tBase', 'tRough', 'tMetal', 'tNormal', 'tHeight', 'tAO'].forEach(function (n) {
  U[n] = gl.getUniformLocation(prog, n);
});

// ---------------------------------------------------------------- mesh upload
var vao = gl.createVertexArray(), bufs = [], indexCount = 0;
function setMesh() {
  var m = buildMesh(shape);
  gl.bindVertexArray(vao);
  bufs.forEach(function (b) { gl.deleteBuffer(b); }); bufs = [];
  function attr(name, data, size) {
    var loc = gl.getAttribLocation(prog, name), b = gl.createBuffer(); bufs.push(b);
    gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  }
  attr('aPos', m.pos, 3); attr('aNor', m.nor, 3); attr('aUv', m.uv, 2);
  var ib = gl.createBuffer(); bufs.push(ib);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(m.idx), gl.STATIC_DRAW);
  indexCount = m.idx.length;
  requestRender();
}

// ---------------------------------------------------------------- textures
var texUnits = ['basecolor', 'roughness', 'metallic', 'normal', 'height', 'ao'];
var uniformOf = {basecolor: 'tBase', roughness: 'tRough', metallic: 'tMetal', normal: 'tNormal', height: 'tHeight', ao: 'tAO'};
var neutral = {basecolor: [128, 128, 128], roughness: [153, 153, 153], metallic: [0, 0, 0],
               normal: [128, 128, 255], height: [128, 128, 128], ao: [255, 255, 255]};
var textures = {};
var aniso = gl.getExtension('EXT_texture_filter_anisotropic');
function setupTexParams(mip) {
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mip ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
  if (mip && aniso) gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT,
    Math.min(8, gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));
}
function solid(rgb) {
  var t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([rgb[0], rgb[1], rgb[2], 255]));
  setupTexParams(false); return t;
}
function fromImage(img) {
  var t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  gl.generateMipmap(gl.TEXTURE_2D); setupTexParams(true); return t;
}
function loadImage(uri) {
  return new Promise(function (resolve, reject) {
    var i = new Image(); i.onload = function () { resolve(i); };
    i.onerror = function () { reject(new Error('could not decode an embedded map')); }; i.src = uri;
  });
}

// ---------------------------------------------------------------- camera / interaction
var rx = 0.25, ry = 0.0, dist = 3.4;
function resetView() { rx = 0.25; ry = 0.0; dist = shape === 'cube' ? 4.2 : (shape === 'plane' ? 3.4 : 3.4); requestRender(); }
var dragging = false, lx = 0, ly = 0;
canvas.addEventListener('pointerdown', function (e) { dragging = true; lx = e.clientX; ly = e.clientY; canvas.classList.add('drag'); canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointerup', function (e) { dragging = false; canvas.classList.remove('drag'); });
canvas.addEventListener('pointermove', function (e) {
  if (!dragging) return;
  ry += (e.clientX - lx) * 0.008; rx += (e.clientY - ly) * 0.008;
  rx = Math.max(-1.45, Math.min(1.45, rx)); lx = e.clientX; ly = e.clientY; requestRender();
});
canvas.addEventListener('wheel', function (e) {
  e.preventDefault(); dist = Math.max(1.8, Math.min(9, dist * (1 + e.deltaY * 0.001))); requestRender();
}, {passive: false});
canvas.addEventListener('dblclick', resetView);

// ---------------------------------------------------------------- render
var lastT = 0;
function draw(t) {
  var w = Math.max(1, Math.floor(canvas.clientWidth * (window.devicePixelRatio || 1)));
  var h = Math.max(1, Math.floor(canvas.clientHeight * (window.devicePixelRatio || 1)));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
  gl.clearColor(0.11, 0.11, 0.125, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.useProgram(prog);
  var model = mul(rotX(rx), rotY(ry)), view = translate(0, 0, -dist), proj = perspective(40 * Math.PI / 180, w / h, 0.1, 50);
  gl.uniformMatrix4fv(U.uProj, false, proj); gl.uniformMatrix4fv(U.uView, false, view); gl.uniformMatrix4fv(U.uModel, false, model);
  var az = values.lightAz * Math.PI / 180, el = values.lightEl * Math.PI / 180;
  gl.uniform3f(U.uLight, Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
  gl.uniform1f(U.uLightI, values.lightI); gl.uniform1f(U.uEnvI, values.envI); gl.uniform1f(U.uTile, values.tile);
  gl.uniform1f(U.uDisp, has('height') ? values.kDisp : 0);
  gl.uniform1f(U.kBase, has('basecolor') ? values.kBase : 0); gl.uniform1f(U.kRough, has('roughness') ? values.kRough : 0);
  gl.uniform1f(U.kMetal, has('metallic') ? values.kMetal : 0); gl.uniform1f(U.kNormal, has('normal') ? values.kNormal : 0);
  gl.uniform1f(U.kAO, has('ao') ? values.kAO : 0); gl.uniform1f(U.kBump, has('height') ? values.kBump : 0);
  gl.uniform2f(U.uTexel, 1 / CFG.size, 1 / CFG.size);
  texUnits.forEach(function (name, i) {
    gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, textures[name]); gl.uniform1i(U[uniformOf[name]], i);
  });
  gl.bindVertexArray(vao);
  gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_INT, 0);
}
function frame(t) {
  var dt = Math.min(0.05, (t - lastT) / 1000 || 0); lastT = t;
  if (auto) { ry += dt * 0.5; dirty = true; }
  if (dirty) { dirty = false; draw(t); }
  requestAnimationFrame(frame);
}
new ResizeObserver(requestRender).observe(canvas);

// ---------------------------------------------------------------- start
buildPanel();
var names = Object.keys(CFG.maps);
Promise.all(names.map(function (n) { return loadImage(CFG.maps[n]); })).then(function (imgs) {
  names.forEach(function (n, i) { textures[n] = fromImage(imgs[i]); });
  texUnits.forEach(function (n) { if (!textures[n]) textures[n] = solid(neutral[n]); });
  setMesh(); resetView();
  var miss = texUnits.filter(function (n) { return !has(n); });
  note.textContent = 'Maps: ' + names.join(', ') + (miss.length ? '  |  not generated: ' + miss.join(', ') : '');
  window.__i2pReady = true;
  requestAnimationFrame(frame);
}).catch(function (e) { note.textContent = 'Error: ' + (e.message || e); console.error(e); });
})();
</script>
</body></html>
'''
