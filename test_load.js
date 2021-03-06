var addon = require(".");
async function main() {
    const options = {
        use_gpu: false,
        gpu_id: 0,
        use_mkldnn: false,
        use_tensorrt: false,
        use_fp16: false,
        gpu_mem: 4000,
        cpu_math_library_num_threads: 10,
        max_side_len: 1920,
        det_db_unclip_ratio: 2.0,
        det_db_box_thresh: 0.5,
        det_db_thresh: 0.3,
    };
    addon.load(
        "./inference/ch_ppocr_mobile_v2.0_det_infer/",
        "./inference/ch_ppocr_mobile_v2.0_rec_infer/",
        "./inference/ppocr_keys_v1.txt",
        options
    );
    console.log("load");
}
main();
