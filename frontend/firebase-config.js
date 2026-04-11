import { initializeApp } from "https://www.gstatic.com/firebasejs/10.10.0/firebase-app.js";
import { getAuth, GoogleAuthProvider } from "https://www.gstatic.com/firebasejs/10.10.0/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/10.10.0/firebase-firestore.js";

const firebaseConfig = {
  apiKey: "AIzaSyBiANROvmbQJviAPhKCzwwTGUEV0BvkMUw",
  authDomain: "florentix-ai.firebaseapp.com",
  projectId: "florentix-ai",
  storageBucket: "florentix-ai.firebasestorage.app",
  messagingSenderId: "973809069371",
  appId: "1:973809069371:web:2e4e10655eea89e7869b1e",
  measurementId: "G-Y2RGXRGGFR"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
const googleProvider = new GoogleAuthProvider();

export { app, auth, db, googleProvider };
