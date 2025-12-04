const navLinks = document.querySelectorAll('.nav-link');
const sections = Array.from(navLinks)
  .map((link) => link.getAttribute('href'))
  .filter((href) => href && href.startsWith('#'))
  .map((href) => document.querySelector(href));
const accountTrigger = document.getElementById('accountTrigger');
const dropdownMenu = document.getElementById('dropdownMenu');
const loginBtn = document.getElementById('loginBtn');
const authModal = document.getElementById('authModal');
const modalOverlay = document.getElementById('modalOverlay');
const modalClose = document.getElementById('modalClose');
const modalTabs = document.querySelectorAll('.modal-tab');
const loginForm = document.getElementById('loginForm');
const signupForm = document.getElementById('signupForm');
const assessmentForm = document.getElementById('assessmentForm');
const assessmentResults = document.getElementById('assessmentResults');
const startAssessment = document.getElementById('startAssessment');
const accountName = document.getElementById('accountName');
const dropdownUser = document.getElementById('dropdownUser');
const currentYear = document.getElementById('currentYear');
const chatToggle = document.getElementById('chatToggle');
const chatWidget = document.getElementById('chatWidget');
const chatClose = document.getElementById('chatClose');
const chatBody = document.getElementById('chatBody');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');

currentYear.textContent = new Date().getFullYear();

// Navigation highlighting
if (sections.length) {
  window.addEventListener('scroll', () => {
    const scrollPos = window.scrollY;
    sections.forEach((section, idx) => {
      if (!section) return;
      const offset = section.offsetTop - 120;
      const height = section.offsetHeight;
      if (scrollPos >= offset && scrollPos < offset + height) {
        navLinks.forEach((link) => link.classList.remove('active'));
        navLinks[idx].classList.add('active');
      }
    });
  });
}

// Modal helpers
const openModal = () => {
  authModal.classList.add('open');
  modalOverlay.classList.add('visible');
  authModal.setAttribute('aria-hidden', 'false');
};

const closeModal = () => {
  authModal.classList.remove('open');
  modalOverlay.classList.remove('visible');
  authModal.setAttribute('aria-hidden', 'true');
};

if (loginBtn) loginBtn.addEventListener('click', openModal);
if (modalClose) modalClose.addEventListener('click', closeModal);
if (modalOverlay) modalOverlay.addEventListener('click', closeModal);

modalTabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    modalTabs.forEach((t) => t.classList.remove('active'));
    tab.classList.add('active');
    const isLogin = tab.dataset.tab === 'login';
    if (loginForm && signupForm) {
      loginForm.classList.toggle('hidden', !isLogin);
      signupForm.classList.toggle('hidden', isLogin);
    }
  });
});

// Fake auth flows
const handleLoginState = (name) => {
  accountName.textContent = name;
  dropdownUser.textContent = name;
  loginBtn.textContent = 'Switch account';
  localStorage.setItem('chmUser', name);
};

if (loginForm) {
  loginForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const email = new FormData(loginForm).get('email');
    handleLoginState((email && email.split('@')[0]) || 'Clinician');
    closeModal();
  });
}

if (signupForm) {
  signupForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const data = new FormData(signupForm);
    const name = data.get('fullName');
    handleLoginState(name || 'New user');
    closeModal();
  });
}

const savedUser = localStorage.getItem('chmUser');
if (savedUser) {
  handleLoginState(savedUser);
}

// Account dropdown interactions
if (accountTrigger && dropdownMenu) {
  accountTrigger.addEventListener('click', (e) => {
    e.stopPropagation();
    dropdownMenu.classList.toggle('open');
  });

  document.addEventListener('click', () => {
    dropdownMenu.classList.remove('open');
  });

  dropdownMenu.addEventListener('click', (e) => {
    e.stopPropagation();
    if (e.target.dataset.action === 'logout') {
      localStorage.removeItem('chmUser');
      handleLoginState('Guest');
      if (loginBtn) loginBtn.textContent = 'Log in';
    }
  });
}

// Smooth scroll to assessment
if (startAssessment) {
  startAssessment.addEventListener('click', () => {
    document.getElementById('assessment').scrollIntoView({ behavior: 'smooth' });
  });
}

// FAQ accordion
document.querySelectorAll('.faq-question').forEach((btn) => {
  btn.addEventListener('click', () => {
    btn.parentElement.classList.toggle('open');
  });
});

// Assessment form submission
const API_URL = 'http://127.0.0.1:5000/predict_gizi';
const CHAT_URL = 'http://127.0.0.1:5000/chat';

const encodeSymptoms = (symptoms) => ({
  kulit_rambut: symptoms.includes('kulit_rambut') ? 1 : 0,
  otot_perut: symptoms.includes('otot_perut') ? 1 : 0,
  imunitas: symptoms.includes('imunitas') ? 1 : 0,
});

const genderEncoding = {
  'Laki-laki': 0,
  Perempuan: 1,
};

const renderResult = (status, body) => {
  assessmentResults.innerHTML = `
    <div class="result-state">
      <span class="result-icon">${status === 'Malnutrisi' ? '⚠️' : '✅'}</span>
      <p class="result-status">${status}</p>
      <p class="result-detail">${body}</p>
    </div>
  `;
};

assessmentForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const age = Number(document.getElementById('ageInput').value);
  const gender = document.getElementById('genderSelect').value;
  const weight = Number(document.getElementById('weightInput').value);
  const height = Number(document.getElementById('heightInput').value);
  const symptoms = Array.from(assessmentForm.querySelectorAll('input[type="checkbox"]:checked')).map(
    (el) => el.value
  );
  const symptomEncoded = encodeSymptoms(symptoms);

  const payload = {
    usia_bulan: age,
    berat_kg: weight,
    tinggi_cm: height,
    jenis_kelamin_encoded: genderEncoding[gender],
    ...symptomEncoded,
  };

  renderResult('Analyzing…', 'Contacting AI model...');

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error('Server error');
    const data = await res.json();
    renderResult(
      data.status_gizi,
      data.status_gizi === 'Malnutrisi'
        ? 'Immediate intervention recommended. Notify clinical supervisor and follow WHO care pathways.'
        : 'Child metrics fall within healthy ranges. Continue regular monitoring and community support.'
    );
  } catch (err) {
    renderResult('Offline', 'Unable to reach API server. Please run `python api_server.py` and retry.');
    console.error(err);
  }
});

// Chatbot interactions
const appendChatMessage = (text, sender) => {
  const div = document.createElement('div');
  div.className = `chat-message ${sender}`;
  div.innerHTML = `<p>${text}</p>`;
  chatBody.appendChild(div);
  chatBody.scrollTop = chatBody.scrollHeight;
};

if (chatToggle && chatWidget) {
  chatToggle.addEventListener('click', () => {
    chatWidget.classList.add('open');
  });
}

if (chatClose && chatWidget) {
  chatClose.addEventListener('click', () => {
    chatWidget.classList.remove('open');
  });
}

if (chatForm && chatInput && chatBody) {
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;

    appendChatMessage(message, 'user');
    chatInput.value = '';

    try {
      const res = await fetch(CHAT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });

      if (!res.ok) throw new Error('Chat server error');
      const data = await res.json();
      appendChatMessage(data.response || 'Maaf, saya tidak mengerti.', 'bot');
    } catch (err) {
      console.error(err);
      appendChatMessage('Tidak dapat menghubungi server chatbot. Pastikan `python app.py` sedang berjalan.', 'bot');
    }
  });
}

// Protect Progress link in navbar on main page
document.querySelectorAll('a[data-protected="true"]').forEach((link) => {
  link.addEventListener('click', (e) => {
    const user = localStorage.getItem('chmUser');
    if (!user && loginBtn) {
      e.preventDefault();
      openModal();
    }
  });
});

// Auto-open login modal when redirected with ?login=1
if (window.location.search.includes('login=1') && loginBtn) {
  openModal();
}