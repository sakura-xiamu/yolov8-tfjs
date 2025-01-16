import React, {useEffect, useRef, useState} from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu"; // set backend to webgl
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import {detect, detectVideo} from "./utils/detect";
import "./style/App.css";

// renderer.js
async function initializeWebGPU() {
    try {
        // 检查 WebGPU 支持
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        // 配置 TensorFlow.js
        await tf.ready();

        // 设置 WebGPU 参数
        tf.env().set('WEBGPU_CPU_FORWARD', false);
        tf.env().set('WEBGPU_FORCE_ASYNC', true);

        // 设置更大的缓冲区
        tf.env().set('WEBGPU_MAX_TEXTURE_DIMENSION_SIZE', 16384);
        tf.env().set('WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_X', 256);

        await tf.setBackend('webgpu');

        console.log('WebGPU initialized successfully');
        console.log('Current backend:', tf.getBackend());
        console.log('Memory state:', tf.memory());
    } catch (error) {
        console.error('WebGPU initialization failed:', error);
        // 降级到 WebGL
        await tf.setBackend('webgl');
    }
}

const App = () => {
    const [loading, setLoading] = useState({loading: true, progress: 0}); // loading state
    const [model, setModel] = useState({
        net: null,
        inputShape: [1, 0, 0, 3],
    }); // init model & input shape

    // references
    const imageRef = useRef(null);
    const cameraRef = useRef(null);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    // model configs
    //const modelName = "yolo11n";
    const modelName = "yolo11s";
    //const modelName = "yolov8n-oiv7";

    useEffect(() => {
        console.log('Available flags:', tf.env().getFlags());
        // 可选：设置其他 WebGPU 参数
        tf.ready().then(async () => {
            // 2. 检查 WebGPU 是否可用
            if (!navigator.gpu) {
                throw new Error('WebGPU is not supported in this browser');
            }
            console.log('navigator.gpu', navigator.gpu)

            // WebGPU 相关的可用配置
            tf.env().set('WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD', 2048);  // 控制 CPU 切换阈值
            tf.env().set('WEBGPU_USE_LOW_POWER_GPU', false);          // 不使用低功耗 GPU
            tf.env().set('WEBGPU_USE_PROFILE_TOOL', false);           // 不使用性能分析工具
            tf.env().set('WEBGPU_CPU_FORWARD', false);                // 禁用 CPU 前向传播
            
            
            // 设置 WebGPU 参数
            tf.setBackend('webgpu').then(async () => {
                
                console.log(`tf.getBackend()=${tf.getBackend()}`)
                console.log('WebGPU 配置:', {
                    cpuHandoffThreshold: tf.env().get('WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD'),
                    useLowPowerGPU: tf.env().get('WEBGPU_USE_LOW_POWER_GPU'),
                    cpuForward: tf.env().get('WEBGPU_CPU_FORWARD')
                });
                // 5. 输出当前状态
                console.log('当前后端:', tf.getBackend());
                console.log('内存状态:', tf.memory());
                const yolov8 = await tf.loadGraphModel(
                    `${window.location.href}/${modelName}_web_model/model.json`,
                    {
                        onProgress: (fractions) => {
                            setLoading({loading: true, progress: fractions}); // set loading fractions
                        },
                    }
                ); // load model

                // warming up model
                const dummyInput = tf.ones(yolov8.inputs[0].shape);
                const warmupResults = yolov8.execute(dummyInput);

                setLoading({loading: false, progress: 1});
                setModel({
                    net: yolov8,
                    inputShape: yolov8.inputs[0].shape,
                }); // set model & input shape
                // 5. 输出当前状态
                console.log('当前后端:', tf.getBackend());
                console.log('内存状态:', tf.memory());
                tf.dispose([warmupResults, dummyInput]); // cleanup memory
            });
        })
    }, []);

    return (
        <div className="App">
            {loading.loading && <Loader>Loading model... {(loading.progress * 100).toFixed(2)}%</Loader>}
            <div className="header">
                <p>
                    模型 : <code className="code">{modelName}</code>
                </p>
            </div>

            <div className="content">
                <img
                    src="#"
                    ref={imageRef}
                    onLoad={() => detect(imageRef.current, model, canvasRef.current)}
                />
                <video
                    autoPlay
                    muted
                    ref={cameraRef}
                    onPlay={() => detectVideo(cameraRef.current, model, canvasRef.current)}
                />
                <video
                    autoPlay
                    muted
                    ref={videoRef}
                    onPlay={() => detectVideo(videoRef.current, model, canvasRef.current)}
                />
                <canvas width={model.inputShape[1]} height={model.inputShape[2]} ref={canvasRef}/>
            </div>

            <ButtonHandler imageRef={imageRef} cameraRef={cameraRef} videoRef={videoRef}/>
        </div>
    );
};

export default App;
