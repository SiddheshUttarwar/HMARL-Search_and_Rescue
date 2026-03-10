const fs = require('fs');
const js = fs.readFileSync('test_script.js', 'utf8');
let bc = 0;
const lines = js.split('\n');
for (let i = 0; i < lines.length; i++) {
  // Very simple stripping of strings/comments to avoid false positives
  let L = lines[i].replace(/\/\/.*$/, ''); // strip single line comments
  // Note: multi-line comments and strings with escaped quotes might trip this up, 
  // but usually it's good enough for a rough estimate in this file.
  
  for (const c of L) {
    if (c === '{') bc++;
    else if (c === '}') {
      bc--;
      if (bc < 0) {
        console.log('ERROR: TOO MANY CLOSING BRACES AT LINE ' + (i+1));
        bc = 0; // reset to keep finding more
      }
    }
  }
}
console.log('FINAL BRACES: ' + bc + ' (Positive means missing closing, negative means too many closing)');
