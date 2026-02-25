function doPost(e) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();

  if (sheet.getLastRow() === 0) {
    sheet.appendRow(["Nume", "Grupa", "Capitol", "Scor", "Procent", "Nota", "Data"]);
  }

  var data = JSON.parse(e.postData.contents);

  sheet.appendRow([
    data.nume,
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

function doGet(e) {
  return ContentService
    .createTextOutput(JSON.stringify({status: "ok", message: "Quiz API active"}))
    .setMimeType(ContentService.MimeType.JSON);
}
