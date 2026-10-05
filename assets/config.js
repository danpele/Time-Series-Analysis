// ============================================================
// TSA site configuration (shared by index.html and index_ro.html)
// GOOGLE_CLIENT_ID: OAuth client (Web) of the Google Cloud project "MFM Quiz Login",
//   shared with the MFM and SFM sites (same origin https://danpele.github.io);
//   the quizzes require "Sign in with Google" with an ASE account (@ase.ro / @stud.ase.ro).
// QUIZ_SCORES_URL: Google Apps Script web app that verifies the Google token and stores
//   the score in the Google Sheet "TSA 2026/2027 - Scoruri quiz" (Apps Script project "TSA Quiz Scores 2026-2027").
// ATTENDANCE_FORM_URL / ATTENDANCE_QR_URL: Google Form "TSA 2026/2027 - Prezență / Attendance" and the QR page
//   of the Apps Script project "TSA Prezenta 2026-2027" (ASE accounts only).
// Values starting with 'YOUR_' are treated as not configured: the site then hides the attendance button,
// shows the quizzes without saving scores, and hides the instructor QR access.
// ============================================================
window.TSA_CONFIG = {
    GOOGLE_CLIENT_ID: '1095360272769-rhjjncfor0gumhev6a0l6tnnrnmdrnna.apps.googleusercontent.com',
    // Apps Script projects "TSA Quiz Scores 2026-2027" and "TSA Prezenta 2026-2027" (Drive: TSA 2026-2027 - Prezență și quiz)
    ATTENDANCE_FORM_URL: 'https://forms.gle/114MpQMCwrM8mbxU8',
    ATTENDANCE_QR_URL: 'https://script.google.com/a/macros/ase.ro/s/AKfycby9J0gXXdHFeOet34ME52-n4nTVQDCXblU4yBdMPJI9PUjlRFoxeFBnyXaXTlePQDIWEw/exec',
    // The QR links appear on the site only after one of these accounts signs in with Google
    // TODO: replace the placeholder with the seminar instructor's ASE address
    INSTRUCTORS: ['danpele@ase.ro', 'YOUR_SEMINAR_INSTRUCTOR_EMAIL'],
    QUIZ_SCORES_URL: 'https://script.google.com/macros/s/AKfycbw3pTvDTI_2jjj7OJ-gejwOGkIEzogX39AP45I_N3jn5tVx6BJeUTpsZ2d2VZ_WL1IjAw/exec'
};
