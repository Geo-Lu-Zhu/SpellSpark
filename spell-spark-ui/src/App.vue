<template>
  <div class="spelling-app">
    <canvas ref="particleCanvas" class="particle-canvas"></canvas>
    <h1>SpellSpark</h1>
    <h1>The Spelling Adventure</h1>
    <div v-if="!isPlaying" class="center-content">
      <button @click="startGame" class="start-btn">Start Game</button>
    </div>

    <div v-else class="game-container">

    <button class="audio-btn" @click="speakWord(currentWord)">
      <img src="/speaker.png" alt="Speaker Icon" class="btn-icon" />
      Hear Word
    </button>

      <div class="input-area">
        <input 
          ref="wordInput"
          v-model="userInput" 
          @keyup.enter="checkSpelling" 
          @keydown.ctrl.prevent="speakWord(currentWord)"
          @input="spawnTypingParticles"
          placeholder="Type here...(Press Ctrl to hear again)" 
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
import { ref, onMounted, onUnmounted, nextTick } from 'vue';
import Papa from 'papaparse';

// --- State Variables ---
const isPlaying = ref(false);
const isAnimating = ref(false);
const vocabulary = ref([]);
const currentWord = ref('');
const definition = ref('');
const userInput = ref('');
const showSuccess = ref(false);
const showFailure = ref(false);

// --- DOM Refs for Canvas & Input ---
const particleCanvas = ref(null);
const wordInput = ref(null);
let ctx = null;
let particles = [];
let animationFrameId = null;

// --- Canvas Particle Logic (Typing Drizzle) ---
onMounted(() => {
  loadVocabulary(); // Load the CSV in the background immediately
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
    p.vy += 0.2; 
    p.alpha -= 0.02; 
    
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
  const rect = wordInput.value.getBoundingClientRect();
  const spawnX = rect.left + Math.random() * rect.width;
  const spawnY = rect.bottom; 

  for (let i = 0; i < 5; i++) {
    particles.push({
      x: spawnX,
      y: spawnY,
      vx: (Math.random() - 0.5) * 4,
      vy: Math.random() * -3, 
      size: Math.random() * 3 + 1,
      color: '255, 165, 0', 
      alpha: 1
    });
  }
};

// --- Data Loading ---
const loadVocabulary = async () => {
  try {
    const response = await fetch('/vocabulary-bank.csv');
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    
    const csvText = await response.text();

    Papa.parse(csvText, {
      header: true, 
      skipEmptyLines: true,
      complete: (results) => {
        const cleanWords = results.data
          .map(row => {
            const rawWord = row.word || row.Word || Object.values(row)[0];
            return rawWord ? String(rawWord).trim() : '';
          })
          .filter(word => word.length > 0);

        vocabulary.value = cleanWords;
      },
      error: (error) => fallbackVocabulary()
    });
  } catch (error) {
    fallbackVocabulary();
  }
};

const fallbackVocabulary = () => {
  vocabulary.value = ['apple', 'banana', 'orange', 'grape', 'answer', 'choose', 'compare'];
};

// --- Audio Logic ---
const speakWord = (word) => {
  if (!word) return;
  // Cancel any stuck speech
  window.speechSynthesis.cancel();
  // Use a global variable to prevent the browser from deleting the speech object
  window.globalUtterance = new SpeechSynthesisUtterance(word);
  window.globalUtterance.volume = 1; 
  window.speechSynthesis.speak(window.globalUtterance);
};

const playApplause = () => {
  const audio = new Audio('/applause.mp3'); 
  audio.volume = 0.5; 
  audio.play().catch(() => {});
  setTimeout(() => { audio.pause(); audio.currentTime = 0; }, 3000);
};

const playMoo = () => {
  const audio = new Audio('/moo.wav');
  audio.volume = 0.6; 
  audio.play().catch(() => {});
  setTimeout(() => { audio.pause(); audio.currentTime = 0; }, 3000);
};

// --- Game Logic ---
const startGame = () => {
  // Unlock the speech engine on the very first click
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(''));
  isPlaying.value = true;
  nextWord();
};

const nextWord = () => {
  userInput.value = '';
  isAnimating.value = false;
  showSuccess.value = false;
  showFailure.value = false;

  if (vocabulary.value.length === 0) fallbackVocabulary();

  const randomIndex = Math.floor(Math.random() * vocabulary.value.length);
  currentWord.value = vocabulary.value[randomIndex];
  definition.value = "A sample definition for the word."; 
  
  // Auto-play the word
  speakWord(currentWord.value);

  // Safely refocus
  setTimeout(() => {
    if (wordInput.value) {
      wordInput.value.disabled = false;
      wordInput.value.focus();
    }
  }, 150);
};

const checkSpelling = () => {
  if (!userInput.value) return;
  isAnimating.value = true; 

  // KEEP AWAKE TRICK: Silently ping the speech engine so it doesn't fall asleep during the timeout
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(''));

  if (userInput.value.toLowerCase().trim() === currentWord.value.toLowerCase()) {
    showSuccess.value = true;
    playApplause(); 
    setTimeout(nextWord, 3000);
  } else {
    showFailure.value = true;
    playMoo(); 
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
  
  /* --- New Width Properties --- */
  width: 100%;
  min-width: 550px; /* Gives plenty of room for the new placeholder */
  max-width: 90vw;  /* Prevents it from breaking off the edge of smaller screens */
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
/* --- White Title with Orange Glow --- */
h1 {
  font-size: 4rem; 
  color: #ffffff; 
  text-transform: uppercase;
  letter-spacing: 3px;
  text-shadow: 
    0 0 10px #FF9800, 
    0 0 20px #FF9800, 
    0 0 40px #E65100; 
  margin-top: 0;
  margin-bottom: 60px; /* Makes the two titles sit nicely together */
}
/* --- Keep the big space below the SECOND title --- */
h1:last-of-type {
  margin-bottom: 120px; /* Pushes the buttons and input box down */
}
/* --- Shared Base for Cartoon Buttons --- */
.start-btn, .audio-btn {
  font-size: 2rem;
  font-weight: 900;
  font-family: 'Comic Sans MS', 'Chalkboard SE', 'Marker Felt', sans-serif;
  padding: 20px 50px; 
  border-radius: 50px; 
  border: 4px solid #ffffff; 
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: 2px;
  outline: none;
  transition: all 0.1s ease-in-out;
}

/* --- Start Game Button (Green Theme) --- */
.start-btn {
  background-color: #4CAF50; /* Bright playful green */
  color: #ffffff;
  box-shadow: 0 10px 0 #2E7D32; /* Dark green 3D shadow */
}

.start-btn:hover {
  background-color: #66BB6A; /* Lighter green on hover */
}

.start-btn:active {
  transform: translateY(10px); 
  box-shadow: 0 0 0 #2E7D32; 
}

/* --- Hear Word Button (Orange Theme) --- */
.audio-btn {
  background-color: #FF9800; /* Vibrant orange */
  color: #ffffff;
  box-shadow: 0 10px 0 #E65100; /* Dark orange 3D shadow */
  /* Increased significantly to push the input box further down */
  margin-bottom: 60px; 
}

.audio-btn:hover {
  background-color: #FFA726; 
}

.audio-btn:active {
  transform: translateY(10px); 
  box-shadow: 0 0 0 #E65100; 
}

/* --- White Input Box with Orange Glow --- */
input {
  background: #1a1a1a; 
  border: 4px solid #ffffff; 
  color: #ffffff; 
  padding: 20px 30px;
  font-size: 2.5rem;
  border-radius: 25px; 
  text-align: center;
  outline: none;
  transition: all 0.3s ease;
  
  /* --- Widened Box --- */
  width: 100%;
  min-width: 750px; /* Increased from 550px to fit the new placeholder */
  max-width: 90vw;  /* Keeps it from breaking mobile screens */
  
  box-shadow: 0 0 15px #FF9800; 
}

input:focus {
  box-shadow: 0 0 25px #FF9800, 0 0 40px #E65100; 
  border-color: #FFB74D; 
}

input::placeholder {
  color: #a0a0a0; 
}
/* --- Shared Base for Cartoon Buttons --- */
.start-btn, .audio-btn {
  font-size: 2rem;
  font-weight: 900;
  font-family: 'Comic Sans MS', 'Chalkboard SE', 'Marker Felt', sans-serif;
  padding: 20px 50px; 
  border-radius: 50px; 
  border: 4px solid #ffffff; 
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: 2px;
  outline: none;
  transition: all 0.1s ease-in-out;
  
  /* --- NEW: Flexbox alignment for the icon --- */
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 15px; /* Adds perfect spacing between the icon and the text */
}

/* --- NEW: Icon Styling --- */
.btn-icon {
  width: 65px; /* Adjust this number to make the icon bigger or smaller */
  height: auto;
  /* Optional: If your PNG is black and you want it to be white to match the text, uncomment the next line */
  /* filter: invert(100%); */ 
}
</style>