// script.js
const BUFFER_SIZE = 100;
const BATCH_SIZE = 8;
const EPOCHS = 3;
const NOISE_DIM = 50;
const NUM_EXAMPLES_TO_GENERATE = 1;
const IMAGE_SIZE = 32;

let generator, discriminator;
let lastGeneratedImageData = null;

// Generar datos dummy ligeros
function generateBatchData() {
    return tf.tidy(() => {
        const data = tf.randomUniform([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3], 0, 1);
        console.log('Batch generado:', data.shape);
        return data;
    });
}

// Construir generador ligero
function buildGenerator() {
    return tf.tidy(() => {
        const model = tf.sequential();
        
        model.add(tf.layers.dense({
            units: 4 * 4 * 64,
            inputShape: [NOISE_DIM],
            useBias: false
        }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.leakyReLU());
        
        model.add(tf.layers.reshape({targetShape: [4, 4, 64]}));
        
        model.add(tf.layers.conv2dTranspose({
            filters: 32,
            kernelSize: 5,
            strides: 2,
            padding: 'same',
            useBias: false
        }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.leakyReLU());
        
        model.add(tf.layers.conv2dTranspose({
            filters: 16,
            kernelSize: 5,
            strides: 2,
            padding: 'same',
            useBias: false
        }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.leakyReLU());
        
        model.add(tf.layers.conv2dTranspose({
            filters: 3,
            kernelSize: 5,
            strides: 2,
            padding: 'same',
            useBias: false,
            activation: 'tanh'
        }));
        
        return model;
    });
}

// Construir discriminador ligero
function buildDiscriminators() {
    return tf.tidy(() => {
        const model = tf.sequential();
        
        model.add(tf.layers.conv2d({
            inputShape: [IMAGE_SIZE, IMAGE_SIZE, 3],
            filters: 16,
            kernelSize: 5,
            strides: 2,
            padding: 'same'
        }));
        model.add(tf.layers.leakyReLU());
        model.add(tf.layers.dropout({rate: 0.3}));
        
        model.add(tf.layers.conv2d({
            filters: 32,
            kernelSize: 5,
            strides: 2,
            padding: 'same'
        }));
        model.add(tf.layers.leakyReLU());
        
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({units: 1}));
        
        return model;
    });
}

// Funciones de pérdida
const crossEntropy = tf.losses.sigmoidCrossEntropy;

function discriminatorLoss(realOutput, fakeOutput) {
    return tf.tidy(() => {
        const realLoss = crossEntropy(tf.onesLike(realOutput), realOutput);
        const fakeLoss = crossEntropy(tf.zerosLike(fakeOutput), fakeOutput);
        return realLoss.add(fakeLoss);
    });
}

function generatorLoss(fakeOutput) {
    return tf.tidy(() => crossEntropy(tf.onesLike(fakeOutput), fakeOutput));
}

// Paso de entrenamiento
async function trainStep(images) {
    return tf.tidy(() => {
        const noise = tf.randomNormal([BATCH_SIZE, NOISE_DIM]);
        const generatedImages = generator.predict(noise);
        
        const realOutput = discriminator.predict(images);
        const fakeOutput = discriminator.predict(generatedImages);
        
        const genLoss = generatorLoss(fakeOutput);
        const discLoss = discriminatorLoss(realOutput, fakeOutput);
        
        const genOptimizer = tf.train.adam(1e-4);
        const genGradients = tf.variableGrads(() => generatorLoss(discriminator.predict(generator.predict(noise))), generator.trainableVariables);
        genOptimizer.applyGradients(genGradients.grads);
        
        const discOptimizer = tf.train.adam(1e-4);
        const discGradients = tf.variableGrads(() => discriminatorLoss(discriminator.predict(images), discriminator.predict(generatedImages)), discriminator.trainableVariables);
        discOptimizer.applyGradients(discGradients.grads);
        
        return {genLoss: genLoss.dataSync()[0], discLoss: discLoss.dataSync()[0]};
    });
}

// Convertir RGB a HSV
function rgbToHsv(r, g, b) {
    r /= 255;
    g /= 255;
    b /= 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;
    let h, s, v = max;
    
    if (delta === 0) h = 0;
    else if (max === r) h = ((g - b) / delta) % 6;
    else if (max === g) h = (b - r) / delta + 2;
    else h = (r - g) / delta + 4;
    h *= 60;
    if (h < 0) h += 360;
    
    s = max === 0 ? 0 : delta / max;
    
    return [h, s, v];
}

// Convertir HSV a RGB
function hsvToRgb(h, s, v) {
    h = (h % 360 + 360) % 360; // Normalizar a 0-360
    const c = v * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = v - c;
    let r, g, b;
    
    if (h < 60) [r, g, b] = [c, x, 0];
    else if (h < 120) [r, g, b] = [x, c, 0];
    else if (h < 180) [r, g, b] = [0, c, x];
    else if (h < 240) [r, g, b] = [0, x, c];
    else if (h < 300) [r, g, b] = [x, 0, c];
    else [r, g, b] = [c, 0, x];
    
    return [
        Math.round((r + m) * 255),
        Math.round((g + m) * 255),
        Math.round((b + m) * 255)
    ];
}

// Aplicar efectos de color con rangos ampliados
function applyColorEffects(imageData, contrast, hueShift, saturation) {
    const result = new Uint8ClampedArray(imageData.length * imageData[0].length * 4);
    for (let i = 0; i < imageData.length; i++) {
        for (let j = 0; j < imageData[i].length; j++) {
            const idx = (i * IMAGE_SIZE + j) * 4;
            let r = (imageData[i][j][0] * 127.5 + 127.5);
            let g = (imageData[i][j][1] * 127.5 + 127.5);
            let b = (imageData[i][j][2] * 127.5 + 127.5);
            
            // Aplicar contraste (rango 0-10)
            r = Math.min(255, Math.max(0, ((r - 128) * contrast + 128)));
            g = Math.min(255, Math.max(0, ((g - 128) * contrast + 128)));
            b = Math.min(255, Math.max(0, ((b - 128) * contrast + 128)));
            
            // Convertir a HSV y ajustar tinte/saturación
            let [h, s, v] = rgbToHsv(r, g, b);
            h = (h + hueShift) % 360; // Rango amplio pero normalizado a 0-360
            s = Math.min(1, Math.max(0, s * saturation)); // Saturación hasta 5, clampada a 0-1
            
            // Volver a RGB
            [r, g, b] = hsvToRgb(h, s, v);
            
            result[idx] = r;
            result[idx + 1] = g;
            result[idx + 2] = b;
            result[idx + 3] = 255;
        }
    }
    return result;
}

// Generar y mostrar imágenes con efectos
function generateAndDisplayImages(contrast = 1.0, hueShift = 0, saturation = 1.0) {
    tf.tidy(() => {
        console.log('Generando imagen...');
        const noise = tf.randomNormal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM]);
        const predictions = generator.predict(noise);
        const imageContainer = document.getElementById('imageContainer');
        
        if (!imageContainer) {
            console.error('Contenedor de imagen no encontrado');
            return;
        }
        
        imageContainer.innerHTML = '';
        
        const canvas = document.createElement('canvas');
        canvas.width = IMAGE_SIZE;
        canvas.height = IMAGE_SIZE;
        const ctx = canvas.getContext('2d');
        
        lastGeneratedImageData = predictions.arraySync()[0];
        console.log('Primer píxel generado:', lastGeneratedImageData[0][0]);
        
        const imgData = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
        const modifiedData = applyColorEffects(lastGeneratedImageData, contrast, hueShift, saturation);
        imgData.data.set(modifiedData);
        
        ctx.putImageData(imgData, 0, 0);
        const scaledCanvas = document.createElement('canvas');
        scaledCanvas.width = 300;
        scaledCanvas.height = 300;
        const scaledCtx = scaledCanvas.getContext('2d');
        scaledCtx.drawImage(canvas, 0, 0, 300, 300);
        imageContainer.appendChild(scaledCanvas);
        console.log('Imagen añadida con contraste:', contrast, 'hue:', hueShift, 'saturación:', saturation);
    });
}

// Función para guardar la imagen generada
function saveImage() {
    const imageContainer = document.getElementById('imageContainer');
    const canvas = imageContainer.querySelector('canvas');
    
    if (!canvas) {
        console.error('No hay imagen generada para guardar');
        return;
    }

    // Crear un enlace temporal para descargar la imagen
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = 'generated_art.png';
    link.click();
}

// Entrenamiento
async function train() {
    const status = document.getElementById('status');
    status.textContent = 'Estado: Entrenando...';
    
    for (let epoch = 0; epoch < EPOCHS; epoch++) {
        const numBatches = Math.ceil(BUFFER_SIZE / BATCH_SIZE);
        for (let i = 0; i < numBatches; i++) {
            const batch = generateBatchData();
            const losses = await trainStep(batch);
            console.log(`Época ${epoch + 1}, Batch ${i + 1}/${numBatches}, Gen Loss: ${losses.genLoss}, Disc Loss: ${losses.discLoss}`);
            batch.dispose();
            await new Promise(resolve => setTimeout(resolve, 10));
        }
    }
    
    status.textContent = 'Estado: Entrenamiento completado';
    generateAndDisplayImages();
}

// Inicializar
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Inicializando modelos...');
    generator = buildGenerator();
    discriminator = buildDiscriminators();
    
    await train();
    
    const generateButton = document.getElementById('generateButton');
    const saveButton = document.getElementById('saveButton'); // Nuevo botón
    const contrastSlider = document.getElementById('contrast');
    const hueSlider = document.getElementById('hue');
    const saturationSlider = document.getElementById('saturation');
    const contrastValue = document.getElementById('contrastValue');
    const hueValue = document.getElementById('hueValue');
    const saturationValue = document.getElementById('saturationValue');
    
    generateButton.addEventListener('click', () => {
        generateAndDisplayImages(
            parseFloat(contrastSlider.value),
            parseFloat(hueSlider.value),
            parseFloat(saturationSlider.value)
        );
    });
    
    // Agregar evento al botón de guardar
    saveButton.addEventListener('click', saveImage);
    
    function updateImage() {
        const contrast = parseFloat(contrastSlider.value);
        const hueShift = parseFloat(hueSlider.value);
        const saturation = parseFloat(saturationSlider.value);
        
        contrastValue.textContent = contrast.toFixed(1);
        hueValue.textContent = `${hueShift}°`;
        saturationValue.textContent = saturation.toFixed(1);
        
        if (lastGeneratedImageData) {
            tf.tidy(() => {
                const imageContainer = document.getElementById('imageContainer');
                imageContainer.innerHTML = '';
                
                const canvas = document.createElement('canvas');
                canvas.width = IMAGE_SIZE;
                canvas.height = IMAGE_SIZE;
                const ctx = canvas.getContext('2d');
                
                const imgData = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
                const modifiedData = applyColorEffects(lastGeneratedImageData, contrast, hueShift, saturation);
                imgData.data.set(modifiedData);
                
                ctx.putImageData(imgData, 0, 0);
                const scaledCanvas = document.createElement('canvas');
                scaledCanvas.width = 300;
                scaledCanvas.height = 300;
                const scaledCtx = scaledCanvas.getContext('2d');
                scaledCtx.drawImage(canvas, 0, 0, 300, 300);
                imageContainer.appendChild(scaledCanvas);
            });
        }
    }
    
    contrastSlider.addEventListener('input', updateImage);
    hueSlider.addEventListener('input', updateImage);
    saturationSlider.addEventListener('input', updateImage);
    
    console.log('Memoria en uso:', tf.memory());
});