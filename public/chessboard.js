function getFenFromUrl() {
  const params = new URLSearchParams(window.location.search);
  return params.get('fen') || 'start';
}

const fenString = getFenFromUrl();
// const fenString = '3R4/6k1/5p2/8/1P1P2p1/2K4p/7B/6r1 b - - 0 1';


const board = Chessboard('myBoard', {
  position: fenString === 'start' ? 'start': fenString,
  draggable: true,
  dropOffBoard: 'snapback',
  pieceTheme: 'assets/{piece}.svg'
});