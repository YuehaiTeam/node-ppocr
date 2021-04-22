var addon = require("./")
const Jimp = require("jimp")

async function main() {
    const image = await Jimp.read("D:\\window.jpg")
    const { width, height, data } = image.bitmap
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
        det_db_thresh: 0.3
    }
    console.log(options)
    addon.load(
        "./inference/ch_ppocr_mobile_v2.0_det_infer/",
        "./inference/ch_ppocr_mobile_v2.0_rec_infer/",
        "./inference/ppocr_keys_v1.txt",
        options
    )
    console.log("load")
    {
        const d = Date.now()
        const p = addon.ocr(width, height, data)
        console.log(Date.now() - d)
        console.log(JSON.stringify(p))
    }
    {
        const d = Date.now()
        const p = addon.ocr(width, height, data)
        console.log(Date.now() - d)
        console.log(JSON.stringify(p))
    }
}
main()
