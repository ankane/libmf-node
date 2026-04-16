import { createHash } from 'node:crypto';
import fs from 'node:fs';
import https from 'node:https';

async function get(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      const chunks = [];
      res.on('data', (chunk) => { chunks.push(chunk); });
      res.on('end', () => {
        if (res.statusCode == 301 || res.statusCode == 302) {
          get(res.headers.location).then(resolve, reject);
        } else {
          resolve(Buffer.concat(chunks));
        }
      });
    }).on('error', (e) => {
      reject(e);
    });
  });
}

async function downloadFile(file, sha256) {
  const url = `https://github.com/ankane/ml-builds/releases/download/libmf-3d5570a/${file}`;
  console.log(`Downloading ${file}...`);
  const contents = await get(url);

  const hash = createHash('sha256');
  const computedHash = hash.update(contents).digest('hex');
  if (computedHash != sha256) {
    throw new Error(`Bad hash: ${computedHash}`);
  }

  const dest = `vendor/${file}`;
  fs.writeFileSync(dest, contents);
  console.log(`Saved ${dest}`);
}

await downloadFile('libmf.so', '2197628cfff98ede7269edc191ec8b7ff6e04edd4d20088938637ddefa596f40');
await downloadFile('libmf.arm64.so', '99d315522ebd118318dad42ffeda08683cbdbd76c5e609cf7a494f9155feca2f');
await downloadFile('libmf.dylib', 'a6ea218370dbb489119e8a561089beea860a05ae0c30e58cc26d5f980d6cb8a2');
await downloadFile('libmf.arm64.dylib', 'fd88da76cb1b9cfdc02fc7dc14a61229195ae9fdf845c78ede7701bb72dfe4e2');
await downloadFile('mf.dll', 'c65eec5ef25482780f8b8f429d55d58ebf494288f84ccb689a2e7e88346fdc40');
