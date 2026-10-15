#!/usr/bin/env node
/**
 * Check the standalone score snapshot against the Virtual Cortex checkout.
 * This script never writes files. To update a reviewed snapshot:
 *   node scripts/data/sync_scoreboard_scores.mjs --patch | apply_patch
 *   perl -pi -e 'chomp if eof' public/data/scoreboard/standardized_results_nsd_1000_models_univariate.json
 *   node scripts/data/sync_scoreboard_scores.mjs --check
 * The formatting step preserves the authoritative JSON file's absent EOF newline.
 */
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const SOURCE_PATH = 'assets/data/new/standardized_results_nsd_1000_models_univariate.json';
const SNAPSHOT_PATH = 'public/data/scoreboard/standardized_results_nsd_1000_models_univariate.json';
const MANIFEST_PATH = 'public/data/scoreboard/source.json';
const ROIS = ['ffa', 'ppa', 'eba'];
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');

export function defaultSourceRoot(repoRoot = REPO_ROOT, exists = fs.existsSync) {
  const containingHost = path.resolve(repoRoot, '../..');
  if (exists(path.join(containingHost, 'package.json'))
    && exists(path.join(containingHost, SOURCE_PATH))) {
    return containingHost;
  }
  return path.resolve(repoRoot, '../cortex-app');
}

function options() {
  const args = process.argv.slice(2);
  let mode = '--check';
  let sourceRoot = defaultSourceRoot();
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === '--check' || args[index] === '--patch') mode = args[index];
    else if (args[index] === '--source-root' && args[index + 1]) sourceRoot = path.resolve(args[++index]);
    else if (args[index] === '--help') {
      console.log('Usage: node scripts/data/sync_scoreboard_scores.mjs [--check|--patch] [--source-root PATH]\nChecks the containing cortex-app checkout when installed at apps/cortex-compass, otherwise the sibling ../cortex-app. An explicit --source-root overrides discovery.\n--patch emits apply_patch text only and never writes files.');
      process.exit(0);
    } else throw new Error(`Unknown or incomplete argument: ${args[index]}`);
  }
  return { mode, sourceRoot };
}

function git(sourceRoot, args) {
  return execFileSync('git', ['-C', sourceRoot, ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
}

function manifest(bytes, sourceCommit) {
  const raw = JSON.parse(bytes.toString('utf8'));
  const inventories = ROIS.map((roi) => {
    if (!raw[roi] || typeof raw[roi] !== 'object' || Array.isArray(raw[roi])) throw new Error(`Source is missing ${roi} data.`);
    for (const [model, data] of Object.entries(raw[roi])) {
      const pair = data?.murty185;
      if (!Array.isArray(pair) || pair.length !== 2 || pair.some((value) => typeof value !== 'number' || !Number.isFinite(value))) {
        throw new Error(`Source ${roi}/${model} lacks a finite Murty185 raw/normalized score pair.`);
      }
    }
    return Object.keys(raw[roi]).filter((model) => model !== 'ceiling').sort();
  });
  if (inventories.some((inventory) => JSON.stringify(inventory) !== JSON.stringify(inventories[0]))) {
    throw new Error('Source model rosters differ across FFA, PPA, and EBA. Review coverage before updating.');
  }
  return {
    sourceRepo: 'https://github.com/murtylab/cortex-app',
    sourcePath: SOURCE_PATH,
    sourceCommit,
    sourceSha256: digest(bytes),
    trainingDataset: 'nsd_1000',
    evaluationDataset: 'murty185',
    metric: 'Univariate Pearson correlation',
    roiKeys: ROIS,
    scorePair: ['raw Pearson correlation', 'stored noise-ceiling-normalized color value'],
    overall: 'Arithmetic mean over the available FFA, PPA, and EBA scores',
    colorPolicy: 'Use stored normalized values, clamped to [0, 1] for the scoreboard palette; never clamp raw scores',
    modelCount: inventories[0].length,
    noiseCeilingKey: 'ceiling',
  };
}

function patchFile(relativePath, nextText) {
  const filename = path.join(REPO_ROOT, relativePath);
  const lines = (text) => text.replace(/\n$/, '').split('\n');
  if (!fs.existsSync(filename)) {
    return `*** Add File: ${relativePath}\n${lines(nextText).map((line) => `+${line}`).join('\n')}\n`;
  }
  const currentText = fs.readFileSync(filename, 'utf8');
  if (currentText === nextText) return '';
  return `*** Update File: ${relativePath}\n@@\n${lines(currentText).map((line) => `-${line}`).join('\n')}\n${lines(nextText).map((line) => `+${line}`).join('\n')}\n`;
}

function runCli() {
  try {
    const { mode, sourceRoot } = options();
    const bytes = fs.readFileSync(path.join(sourceRoot, SOURCE_PATH));
    if (mode === '--patch') {
      const commit = git(sourceRoot, ['rev-parse', 'HEAD']).toString('utf8').trim();
      if (!git(sourceRoot, ['show', `${commit}:${SOURCE_PATH}`]).equals(bytes)) {
        throw new Error('The source score file has uncommitted changes. Commit/review it before recording provenance.');
      }
      const nextManifest = `${JSON.stringify(manifest(bytes, commit), null, 2)}\n`;
      const snapshotEdits = patchFile(SNAPSHOT_PATH, bytes.toString('utf8'));
      const edits = snapshotEdits + patchFile(MANIFEST_PATH, nextManifest);
      console.log(`*** Begin Patch\n${edits}*** End Patch`);
      if (snapshotEdits && !bytes.toString('utf8').endsWith('\n')) {
        console.error('After applying the patch, remove the snapshot EOF newline with the formatting command documented at the top of this script, then run --check.');
      }
    } else {
      const snapshot = fs.readFileSync(path.join(REPO_ROOT, SNAPSHOT_PATH));
      const provenance = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, MANIFEST_PATH), 'utf8'));
      if (!snapshot.equals(bytes)) {
        throw new Error('The standalone scoreboard snapshot differs from cortex-app. Review --patch output before applying it (including the documented EOF formatting step).');
      }
      if (!/^[a-f0-9]{40}$/.test(provenance.sourceCommit)) throw new Error('The source manifest does not record a valid commit.');
      const expected = manifest(bytes, provenance.sourceCommit);
      if (JSON.stringify(provenance) !== JSON.stringify(expected)) throw new Error('The source manifest differs from the reviewed scoreboard source/context.');
      if (!git(sourceRoot, ['show', `${provenance.sourceCommit}:${SOURCE_PATH}`]).equals(bytes)) {
        throw new Error('The recorded source commit does not contain the reviewed scoreboard bytes.');
      }
      console.log(`Scoreboard snapshot matches cortex-app: ${expected.modelCount} models + noise ceiling, 3 ROIs, NSD-1000 → Murty185, Pearson r. SHA256 ${expected.sourceSha256}`);
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) runCli();
