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
  const url = `https://github.com/ankane/ml-builds/releases/download/libmf-master-2/${file}`;
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

await downloadFile('libmf.so', '5a22ec277a14ab8e3b8efacfec7fe57e5ac4192ea60e233d7e6db38db755a67e');
await downloadFile('libmf.arm64.so', '223ef5d1213b883c8cb8623bf07bf45167cd48585a5f2b59618cea034c72ad61');
await downloadFile('libmf.dylib', '6e3451feeded62a2e761647aef7c2a0e7dbeeee83ce8d4ab06586f5820f7ebf9');
await downloadFile('libmf.arm64.dylib', '063c1dc39a6fda12ea2616d518fa319b8ab58faa65b174f176861cf8f8eaae0d');
await downloadFile('mf.dll', '8b0e53ab50ca3e2b365424652107db382dff47a26220c092b89729f9c3b8d7e7');
