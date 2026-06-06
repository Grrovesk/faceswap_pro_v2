"""v2 webcam tab builder. Slim version of legacy 705-line tab.

Hard constraint: zero imports from ui.app or ui.stream_server.

Layout (3 col):
  COLUMN 1 - Source face + camera settings
  COLUMN 2 - Live MJPEG <img> + Start/Stop buttons + stats panel
  COLUMN 3 - Status / log

Features included this session:
  - Source face upload (gr.File, like Face Swap tab does it well)
  - Camera device index, resolution, JPEG quality
  - GPU id, detection threshold, blend method
  - MJPEG <img> served from /webcam_stream/video
  - Start/Stop buttons that POST to /webcam_stream/start, /stop
  - Stats panel polling /webcam_stream/stats every 1s

Features NOT in this session (add in follow-up if needed):
  - Background replacement (rembg / chroma key)
  - Brightness / contrast sliders
  - Recording-to-disk
  - Music soundtrack player
  - Character mode (animal / monster)
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from . import state as webcam_state


# Same JS snippet legacy uses (verbatim): start/stop hit the FastAPI
# routes registered by streaming.register_stream_routes.
_HTML_BLOCK = """
<div style='text-align:center'>
  <img id='fsp_v2_stream' src=''
       style='max-width:100%;border-radius:6px;
              background:#111;min-height:360px;
              width:100%;' />
  <div style='margin-top:8px'>
    <button id='fsp_v2_start'
      style='padding:8px 18px;margin-right:8px;font-weight:bold;
             background:#2a7;color:white;border:0;border-radius:4px;'
      onclick="
        var dn=document.querySelector('#wc_device input');
        var wn=document.querySelector('#wc_width input');
        var hn=document.querySelector('#wc_height input');
        var qn=document.querySelector('#wc_quality input');
        var d=parseInt((dn&&dn.value)||0);
        var w=parseInt((wn&&wn.value)||640);
        var h=parseInt((hn&&hn.value)||480);
        var q=parseInt((qn&&qn.value)||80);
        var url='/webcam_stream/start?device='+d+'&width='+w+'&height='+h+'&quality='+q;
        fetch(url,{method:'POST'}).then(function(r){return r.json();})
          .then(function(j){
            if (!j.running && j.error) {
              alert('camera failed: ' + j.error);
              return;
            }
            var img=document.getElementById('fsp_v2_stream');
            img.src='/webcam_stream/video?t='+Date.now();
          });">▶ Start camera</button>
    <button id='fsp_v2_stop'
      style='padding:8px 18px;font-weight:bold;
             background:#a33;color:white;border:0;border-radius:4px;'
      onclick="fetch('/webcam_stream/stop',{method:'POST'}).then(function(){
        var img=document.getElementById('fsp_v2_stream'); img.src='';
        /* Backend's stop() already calls stop_recording() internally.
           Reset the Record button visual state so it doesn't stay
           stuck on '⏺ Recording...' after the camera stops. */
        var rb=document.getElementById('fsp_v2_rec');
        if (rb) { rb.style.background='#c44';
                  rb.textContent='⏺ Record'; }
      });">■ Stop</button>
    <button id='fsp_v2_rec'
      style='padding:8px 18px;margin-left:16px;font-weight:bold;
             background:#c44;color:white;border:0;border-radius:4px;'
      onclick="fetch('/webcam_stream/record/start',{method:'POST'})
        .then(function(r){return r.json();})
        .then(function(j){
          if (j.recording) {
            this_btn=document.getElementById('fsp_v2_rec');
            this_btn.style.background='#770';
            this_btn.textContent='⏺ Recording...';
          } else if (j.error) {
            alert('record failed: ' + j.error);
          }
        });">⏺ Record</button>
    <button id='fsp_v2_rec_stop'
      style='padding:8px 18px;margin-left:8px;font-weight:bold;
             background:#444;color:white;border:0;border-radius:4px;'
      onclick="fetch('/webcam_stream/record/stop',{method:'POST'})
        .then(function(r){return r.json();})
        .then(function(j){
          var rb=document.getElementById('fsp_v2_rec');
          rb.style.background='#c44';
          rb.textContent='⏺ Record';
          if (j.saved) {
            var nm=j.path.split(/[\\/]/).pop();
            _fspMsg('Saved: ' + nm, '#4a4');
          }
        });">⏹ Stop record</button>
    <button id='fsp_v2_open_dir'
      style='padding:8px 18px;margin-left:16px;font-weight:bold;
             background:#356;color:white;border:0;border-radius:4px;'
      onclick="fetch('/webcam_stream/open_recordings_dir',{method:'POST'})
        .then(function(r){return r.json();})
        .then(function(j){
          if (j.ok) {
            _fspMsg('Opened folder', '#4a4');
          } else {
            _fspMsg('Could not open: ' + (j.error||''), '#c44');
          }
        });">📁 Open recordings folder</button>
    <span id='fsp_v2_msg'
      style='margin-left:16px;font-family:monospace;font-size:13px;
             transition:opacity 0.5s;opacity:0;'></span>
  </div>
  <div id='fsp_v2_stats'
       style='margin-top:10px;font-family:monospace;
              color:#888;font-size:13px;display:none;'></div>
</div>
<script>
(function(){
  if (window.fsp_v2_polling) return;
  window.fsp_v2_polling = true;
  /* Transient inline status helper used by Stop-record + Open-folder. */
  window._fspMsg = function(text, color){
    var m = document.getElementById('fsp_v2_msg');
    if (!m) return;
    m.textContent = text;
    m.style.color = color || '#4a4';
    m.style.opacity = '1';
    if (window._fspMsgTimer) clearTimeout(window._fspMsgTimer);
    window._fspMsgTimer = setTimeout(function(){
      m.style.opacity = '0';
    }, 4000);
  };
  function poll(){
    fetch('/webcam_stream/stats').then(function(r){return r.json();})
      .then(function(j){
        var el=document.getElementById('fsp_v2_stats');
        if (!el) return;
        if (j.running) {
          el.style.display = '';
          el.textContent = 'running  fps='+j.fps+'  latency='+j.latency_ms+'ms  frames='+j.frames;
          el.style.color = '#4a4';
        } else if (j.error) {
          el.style.display = '';
          el.textContent = 'stopped  -- '+j.error;
          el.style.color = '#c44';
        } else {
          /* nothing running, no error -- hide the stats line entirely
             instead of leaving "(not running)" hanging under the
             stream area. */
          el.style.display = 'none';
          el.textContent = '';
        }
      }).catch(function(){});
  }
  setInterval(poll, 1000); poll();
})();
</script>
"""


def _on_source_change(source_file):
    """Push the new source path into webcam_state. The next captured
    frame's swap_fn will pick it up."""
    p = getattr(source_file, "name", source_file)
    webcam_state.set_source(str(p) if p else None)
    if p and Path(str(p)).is_file():
        return str(p), f"source set: `{Path(str(p)).name}`"
    return None, "(no source)"


def _on_options_change(gpu_id, det_threshold, blend_method,
                         brightness, contrast, saturation,
                         virtual_cam_on):
    webcam_state.set_options(
        gpu_id=int(gpu_id),
        det_threshold=float(det_threshold),
        blend_method=str(blend_method),
        brightness=int(brightness),
        contrast=float(contrast),
        saturation=float(saturation),
        virtual_cam_on=bool(virtual_cam_on),
    )
    s = webcam_state.get_snapshot()
    msg = (f"opts: gpu={s.gpu_id} thresh={s.det_threshold:.2f} "
           f"blend={s.blend_method}  |  "
           f"b={s.brightness:+d} c={s.contrast:.2f} sat={s.saturation:.2f}")
    if s.virtual_cam_on:
        try:
            from . import virtual_cam as _vc
            msg += f"  |  vcam: {_vc.status()}"
        except Exception:
            msg += "  |  vcam: import failed"
    return msg


def build_tab() -> None:
    """Build the Webcam tab body. Caller is responsible for wrapping
    in `with gr.Tab(...)` first."""
    with gr.Row(equal_height=False):
        # COLUMN 1: source + options
        with gr.Column(scale=1):
            gr.Markdown("### 1. Source face")
            # file_types=["image"] DELIBERATELY OMITTED: gr.File's
            # preprocess validates against file_types and on rejection
            # caches the invalid path in component state -- every
            # subsequent upload (even valid ones) then re-raises
            # "Invalid file type" forever. Downstream InsightFace
            # face detection cleanly rejects non-images with a clear
            # error; that's better UX than a wedged UI.
            wc_source = gr.File(
                label="Source face image (.jpg / .png)",
                type="filepath")
            wc_source_preview = gr.Image(
                interactive=False, height=200,
                type="filepath", show_label=False,
                show_download_button=False, container=False)
            wc_source_status = gr.Markdown("(no source)")

            gr.Markdown("### 2. Camera + swap options")
            with gr.Group():
                wc_device = gr.Number(
                    value=0, precision=0,
                    label="Camera device index",
                    info="0 = default/built-in. Try 1, 2 if you have "
                         "multiple webcams. Read at Start camera click.",
                    elem_id="wc_device")
                wc_width = gr.Number(
                    value=640, precision=0,
                    label="Width (px)",
                    info="Capture width. Camera HW may ignore unsupported "
                         "values and return native res. Read at Start "
                         "camera click -- Stop and Start to apply changes.",
                    elem_id="wc_width")
                wc_height = gr.Number(
                    value=480, precision=0,
                    label="Height (px)",
                    info="Capture height. Same caveat as Width. "
                         "Read at Start camera click.",
                    elem_id="wc_height")
                wc_quality = gr.Slider(
                    40, 100, value=80, step=5,
                    label="JPEG quality",
                    info="Compression of the live MJPEG stream sent to "
                         "the browser. 80 = default. Higher = sharper "
                         "preview, more bandwidth. Has NO effect on the "
                         "saved recording or OBS virtual cam output. "
                         "Read at Start camera click.",
                    elem_id="wc_quality")
                wc_gpu = gr.Number(value=0, precision=0,
                                    label="GPU id")
                wc_thresh = gr.Slider(0.1, 0.9, value=0.5, step=0.05,
                                       label="Detection threshold")
                wc_blend = gr.Dropdown(
                    choices=["poisson", "alpha", "feather", "none"],
                    value="poisson", label="Blend method")

            # Brightness / Contrast / Saturation MOVED to column 2 so
            # the user can see the live feed while adjusting them.
            # Buried-in-column-1 caused the feed to scroll offscreen
            # during tuning.

            gr.Markdown("### 4. Streaming output")
            with gr.Group():
                wc_vcam = gr.Checkbox(
                    label="Send to virtual camera (OBS / Zoom / Discord)",
                    value=False,
                    info="When ON, each filtered+swapped frame goes "
                         "to the OBS virtual camera driver. On "
                         "Windows install OBS Studio first (provides "
                         "the driver). pyvirtualcam must be installed: "
                         "pip install pyvirtualcam")
            wc_opts_status = gr.Markdown("")

        # COLUMN 2/3 merged: the live feed + the 3 sliders the user
        # actually tunes against what they see on screen.
        with gr.Column(scale=2):
            gr.Markdown("### 3. Live feed")
            gr.HTML(_HTML_BLOCK)
            with gr.Accordion("Output filters (post-swap)",
                              open=False):
                wc_brightness = gr.Slider(
                    -100, 100, value=0, step=1,
                    label="Brightness",
                    info="-100 = darker, 0 = none, +100 = brighter")
                wc_contrast = gr.Slider(
                    0.5, 2.0, value=1.0, step=0.05,
                    label="Contrast",
                    info="1.0 = none; <1 flatter, >1 punchier")
                wc_saturation = gr.Slider(
                    0.0, 2.0, value=1.0, step=0.05,
                    label="Saturation",
                    info="0 = black & white, 1 = none, 2 = vivid")

    # Wiring -- pure data flow, no business logic
    wc_source.change(
        fn=_on_source_change,
        inputs=[wc_source],
        outputs=[wc_source_preview, wc_source_status])
    _opt_inputs = [wc_gpu, wc_thresh, wc_blend,
                   wc_brightness, wc_contrast, wc_saturation, wc_vcam]
    for opt in _opt_inputs:
        opt.change(
            fn=_on_options_change,
            inputs=_opt_inputs,
            outputs=[wc_opts_status])
