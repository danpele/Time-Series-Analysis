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
    // TODO: deploy the TSA Apps Scripts (quiz scores, attendance) and paste their URLs here
    ATTENDANCE_FORM_URL: 'YOUR_ATTENDANCE_FORM_URL',
    ATTENDANCE_QR_URL: 'YOUR_ATTENDANCE_QR_URL',
    // The QR links appear on the site only after one of these accounts signs in with Google
    // TODO: replace the placeholder with the seminar instructor's ASE address
    INSTRUCTORS: ['danpele@ase.ro', 'YOUR_SEMINAR_INSTRUCTOR_EMAIL'],
    QUIZ_SCORES_URL: 'YOUR_QUIZ_SCORES_URL'
};
