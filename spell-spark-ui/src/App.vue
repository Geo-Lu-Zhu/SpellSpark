<template>
  <div class="spelling-app">
    <canvas ref="particleCanvas" class="particle-canvas"></canvas>

    <h1>Spelling Adventure</h1>

    <div v-if="!isPlaying" class="center-content">
      <button @click="startGame" class="start-btn">Start Game</button>
    </div>

    <div v-else class="game-container">
      
      <div class="definition-box" v-if="definition">
        <p><strong>Meaning:</strong> {{ definition }}</p>
      </div>

      <button @click="speakWord(currentWord)" class="audio-btn">🔊 Hear Word</button>

      <div class="input-area">
        <input 
          ref="wordInput"
          v-model="userInput" 
          @keyup.enter="checkSpelling" 
          @input="spawnTypingParticles"
          placeholder="Type here..." 
          :disabled="isAnimating"
          autofocus
        />
      </div>

      <div v-if="showSuccess" class="overlay">
        <h2 class="correct-word-fire glow-pulse">{{ currentWord }}</h2>
      </div>

      <div v-if="showFailure" class="overlay failure-layout">
        <h2 class="wrong-word smoke-fade">{{ userInput }}</h2>
        <h2 class="correct-word-fire show-delayed">{{ currentWord }}</h2>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import Papa from 'papaparse';

// State Variables
const isPlaying = ref(false);
const isAnimating = ref(false);
const vocabulary = ref([]);
const currentWord = ref('');
const definition = ref('');
const userInput = ref('');
const showSuccess = ref(false);
const showFailure = ref(false);

// DOM Refs for Canvas & Input
const particleCanvas = ref(null);
const wordInput = ref(null);
let ctx = null;
let particles = [];
let animationFrameId = null;

// --- Canvas Particle Logic (Typing Drizzle) ---
onMounted(() => {
  if (particleCanvas.value) {
    ctx = particleCanvas.value.getContext('2d');
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    animateParticles();
  }
});

onUnmounted(() => {
  window.removeEventListener('resize', resizeCanvas);
  cancelAnimationFrame(animationFrameId);
});

const resizeCanvas = () => {
  particleCanvas.value.width = window.innerWidth;
  particleCanvas.value.height = window.innerHeight;
};

const animateParticles = () => {
  ctx.clearRect(0, 0, particleCanvas.value.width, particleCanvas.value.height);
  
  for (let i = particles.length - 1; i >= 0; i--) {
    let p = particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.vy += 0.2; // Gravity
    p.alpha -= 0.02; // Fade out
    
    ctx.fillStyle = `rgba(${p.color}, ${p.alpha})`;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
    ctx.fill();
    
    if (p.alpha <= 0) {
      particles.splice(i, 1);
    }
  }
  animationFrameId = requestAnimationFrame(animateParticles);
};

const spawnTypingParticles = () => {
  if (!wordInput.value) return;
  
  // Get input position to spawn particles nearby
  const rect = wordInput.value.getBoundingClientRect();
  const spawnX = rect.left + Math.random() * rect.width;
  const spawnY = rect.bottom; // Drizzle from the bottom of the input

  for (let i = 0; i < 5; i++) {
    particles.push({
      x: spawnX,
      y: spawnY,
      vx: (Math.random() - 0.5) * 4, // Spread left/right
      vy: Math.random() * -3, // Slight upward burst before falling
      size: Math.random() * 3 + 1,
      color: '255, 165, 0', // Orange RGB
      alpha: 1
    });
  }
};

// --- Game Logic ---
const loadVocabulary = async () => {
  // Simplified for example; replace with your actual fetch logic
  vocabulary.value = ['apple', 'banana', 'orange', 'grape'];
};

const startGame = async () => {
  await loadVocabulary();
  isPlaying.value = true;
  nextWord();
};

const nextWord = async () => {
  userInput.value = '';
  isAnimating.value = false;
  showSuccess.value = false;
  showFailure.value = false;

  const randomIndex = Math.floor(Math.random() * vocabulary.value.length);
  currentWord.value = vocabulary.value[randomIndex];
  definition.value = "A sample definition for the word."; 
  
  speakWord(currentWord.value);
  // Re-focus input after resetting
  setTimeout(() => wordInput.value && wordInput.value.focus(), 100);
};

const speakWord = (word) => {
  const utterance = new SpeechSynthesisUtterance(word);
  window.speechSynthesis.speak(utterance);
};

const playApplause = () => {
  const audio = new Audio('/applause.mp3'); 
  audio.volume = 0.5; 
  
  audio.play().catch(e => console.log("Audio play blocked by browser:", e));

  // Stop the audio after 3000 milliseconds (3 seconds)
  setTimeout(() => {
    audio.pause();           // Stop the playback
    audio.currentTime = 0;   // Rewind back to the very beginning
  }, 3000);
};

const playMoo = () => {
  const audio = new Audio('/moo.wav');
  audio.volume = 0.6; 
  
  audio.play().catch(e => console.log("Audio play blocked by browser:", e));

  // Stop the audio after 3000 milliseconds (3 seconds)
  setTimeout(() => {
    audio.pause();           // Stop the playback
    audio.currentTime = 0;   // Rewind back to the very beginning
  }, 3000);
};

const checkSpelling = () => {
  if (!userInput.value) return;
  isAnimating.value = true; 

  if (userInput.value.toLowerCase().trim() === currentWord.value.toLowerCase()) {
    // Correct Answer
    showSuccess.value = true;
    playApplause(); // Play the applause sound
    
    setTimeout(nextWord, 3000);
  } else {
    // Wrong Answer
    showFailure.value = true;
    playMoo(); // Play the cow moo sound
    
    setTimeout(nextWord, 5000); 
  }
};
</script>

<style scoped>
/* Base Dark Theme Setup */
.spelling-app {
  background-color: #0d0d0d;
  color: #ffffff;
  min-height: 100vh;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 50px;
  position: relative;
  overflow: hidden;
}

/* Canvas needs to cover everything but let clicks pass through */
.particle-canvas {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  pointer-events: none; 
  z-index: 100;
}

.game-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  z-index: 10;
}

input {
  background: #1a1a1a;
  border: 2px solid #333;
  color: #fff;
  padding: 15px 25px;
  font-size: 2rem;
  border-radius: 10px;
  text-align: center;
  outline: none;
  transition: border-color 0.3s;
}

input:focus {
  border-color: #ff8c00;
}

/* --- Animations --- */
.overlay {
  position: absolute;
  top: 60%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 100%;
}

.failure-layout {
  gap: 20px;
}

/* 1. Smoke Fade (Wrong Word) */
.smoke-fade {
  color: #a0a0a0;
  font-size: 3rem;
  margin: 0;
  animation: smokeDissolve 3s forwards;
}

@keyframes smokeDissolve {
  0% { opacity: 1; filter: blur(0px); transform: translateY(0) scale(1); }
  50% { opacity: 0.8; filter: blur(4px); transform: translateY(-30px) scale(0.8); }
  100% { opacity: 0; filter: blur(15px); transform: translateY(-80px) scale(0.5); }
}

/* 2. Glowing Fire Effect (Correct Word) */
.correct-word-fire {
  font-size: 6rem;
  margin: 0;
  color: #fff;
  text-transform: uppercase;
  letter-spacing: 5px;
  text-shadow: 
    0 0 5px #fff, 
    0 0 10px #ffea00, 
    0 0 20px #ff9d00, 
    0 0 40px #ff2a00, 
    0 0 80px #ff0000;
}

.glow-pulse {
  animation: pulsateGlow 1.5s infinite alternate;
}

.show-delayed {
  opacity: 0;
  animation: revealFire 1.5s 1s forwards, pulsateGlow 1.5s 1s infinite alternate;
}

@keyframes pulsateGlow {
  0% { text-shadow: 0 0 5px #fff, 0 0 10px #ffea00, 0 0 20px #ff9d00, 0 0 40px #ff2a00; }
  100% { text-shadow: 0 0 10px #fff, 0 0 20px #ffea00, 0 0 40px #ff9d00, 0 0 80px #ff2a00, 0 0 100px #ff0000; }
}

@keyframes revealFire {
  0% { opacity: 0; transform: scale(0.5); }
  100% { opacity: 1; transform: scale(1); }
}
</style>