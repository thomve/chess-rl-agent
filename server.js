const express = require('express');
const app = express();
const path = require('path');

const PORT = 3000;

app.get('/chessboard', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'chessboard.html'));
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});