// appsscript.json manifest (set in Project Settings):
// {
//   "timeZone": "Europe/Bucharest",
//   "dependencies": {},
//   "exceptionLogging": "STACKDRIVER",
//   "oauthScopes": [
//     "https://www.googleapis.com/auth/spreadsheets",
//     "https://www.googleapis.com/auth/script.external_request"
//   ]
// }

// Configuration
var GITHUB_CLIENT_ID     = 'Ov23liJNDMSO6UzqPjZp';
var GITHUB_CLIENT_SECRET = 'bbddf00768f1c2d91a58c970aa8e16a7968c0e1b';

// POST handler - quiz results
function doPost(e) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();

  if (sheet.getLastRow() === 0) {
    sheet.appendRow(["Nume", "GitHub Username", "Grupa", "Capitol", "Scor", "Procent", "Nota", "Data"]);
  }

  var data = JSON.parse(e.postData.contents);

  sheet.appendRow([
    data.nume,
    data.github_username || '',
    data.grupa,
    data.capitol,
    data.scor,
    data.total,
    data.nota,
    new Date().toLocaleString("ro-RO", {timeZone: "Europe/Bucharest"})
  ]);

  return ContentService
    .createTextOutput(JSON.stringify({status: "ok"}))
    .setMimeType(ContentService.MimeType.JSON);
}

// GET handler - status check + GitHub OAuth exchange
function doGet(e) {
  var params = e.parameter || {};

  if (params.action === 'github_auth' && params.code) {
    return handleGitHubAuth(params.code);
  }

  return ContentService
    .createTextOutput(JSON.stringify({status: "ok", message: "Quiz API active"}))
    .setMimeType(ContentService.MimeType.JSON);
}

// GitHub OAuth helper
function handleGitHubAuth(code) {
  try {
    var tokenResp = UrlFetchApp.fetch('https://github.com/login/oauth/access_token', {
      method: 'post',
      headers: { 'Accept': 'application/json' },
      payload: {
        client_id: GITHUB_CLIENT_ID,
        client_secret: GITHUB_CLIENT_SECRET,
        code: code
      }
    });

    var tokenData = JSON.parse(tokenResp.getContentText());

    if (tokenData.error) {
      return jsonResponse({ error: tokenData.error_description || tokenData.error });
    }

    var userResp = UrlFetchApp.fetch('https://api.github.com/user', {
      headers: {
        'Authorization': 'Bearer ' + tokenData.access_token,
        'Accept': 'application/json',
        'User-Agent': 'TSA-Quiz-App'
      }
    });

    var user = JSON.parse(userResp.getContentText());

    return jsonResponse({
      login: user.login,
      name: user.name || user.login,
      avatar_url: user.avatar_url
    });

  } catch (err) {
    return jsonResponse({ error: 'Authentication failed: ' + err.message });
  }
}

function jsonResponse(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
