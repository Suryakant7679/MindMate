// Browser check of the analytics dashboard against the backend records.
//   node tests/browser/dashboard_check.js [base url]      (needs Playwright; set PLAYWRIGHT_PATH to its node_modules folder)
const path = process.env.PLAYWRIGHT_PATH || '/tmp/mindmate-browser/node_modules';
const { chromium } = require(path + '/playwright');
const fs = require('fs');
const BASE = process.argv[2] || 'http://127.0.0.1:8093';
const OUT = 'results/browser_check';
const MESSAGES = [
  'Everyone forgot about me today and it really hurts. I feel so alone.',
  'yaar aaj office mein boss ne sabke saamne daant diya, bahut akela feel ho raha hai',
  'I keep worrying that I will lose my job and let my family down.',
  'thanks, that helped. I just found out I passed my exam and I am so happy!',
];
const TABS = ['Overview', 'Emotion Timeline', 'Response Comparison', 'Neuron Explorer', 'Causal Ablations', 'Model Evaluation'];
const get = async (page, url) => page.evaluate(async (u) => (await fetch(u)).json(), url);
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1360, height: 900 } });
  const errors = [], report = { tabs: {}, checks: [] };
  const check = (name, ok, detail) => { report.checks.push({ name, ok: !!ok, detail }); console.log(ok ? 'ok  ' : 'FAIL', name, detail ?? ''); };
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  let conversation = null, lastTurn = null;
  page.on('request', (request) => {
    if (request.url().endsWith('/api/chat/stream')) conversation = JSON.parse(request.postData()).conversation_id;
    const match = request.url().match(/\/api\/emotion\/trace\/([0-9a-f]{32})/); if (match) lastTurn = match[1];
  });
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  // Empty state before any conversation.
  await page.getByRole('button', { name: /Mood Tracker/ }).click();
  await page.waitForTimeout(1200);
  for (const name of TABS) {
    await page.getByRole('tab', { name }).click(); await page.waitForTimeout(500);
    report.tabs['empty:' + name] = (await page.locator('#mood-panel').innerText()).slice(0, 160);
  }
  check('empty states render on every tab without a conversation', Object.values(report.tabs).every((text) => text.length > 20));
  await page.screenshot({ path: `${OUT}/empty_overview.png` });
  await page.getByRole('button', { name: /^Chat/ }).click();
  for (const text of MESSAGES) {
    await page.getByRole('textbox', { name: 'Message', exact: true }).fill(text);
    await page.getByLabel('Send message').click();
    await page.waitForSelector('.thinking', { state: 'attached', timeout: 60000 }).catch(() => {});
    await page.waitForSelector('.thinking', { state: 'detached', timeout: 600000 });
    await page.waitForTimeout(2000);
  }
  await page.getByRole('button', { name: /Mood Tracker/ }).click();
  await page.waitForTimeout(2500);
  const view = await get(page, `/api/emotion/conversation/${conversation}`);
  const models = await get(page, '/api/emotion/models');
  const research = await get(page, '/api/emotion/research');
  const lookup = (pathText) => pathText.split('.').reduce((node, part) => part === 'length' ? node.length : node[part], view);
  let compared = 0, mismatched = [];
  for (const name of TABS) {
    await page.getByRole('tab', { name }).click();
    await page.waitForTimeout(1200);
    const panel = page.locator('#mood-panel');
    const metrics = await panel.locator('[data-metric]').evaluateAll((els) => els.map((e) => ({ path: e.dataset.metric, value: e.dataset.value })));
    for (const metric of metrics) {
      if (metric.value === '') continue;
      const raw = lookup(metric.path); compared += 1;
      if (Math.abs(Number(metric.value) - Number(raw)) > 1e-9) mismatched.push({ tab: name, ...metric, raw });
    }
    report.tabs[name] = { headings: await panel.locator('h2').allTextContents(), charts: await panel.locator('.recharts-responsive-container').count(),
      badges: [...new Set(await panel.locator('.badge.source').allTextContents())], metrics: metrics.length };
    await page.screenshot({ path: `${OUT}/${name.replace(/\s+/g, '_').toLowerCase()}.png`, fullPage: true });
  }
  check(`marked values equal the conversation record (${compared} compared)`, compared > 20 && mismatched.length === 0, JSON.stringify(mismatched.slice(0, 3)));
  check('conversation has the four turns sent', view.trajectory.turns.length === MESSAGES.length, view.trajectory.turns.length);

  // Response Comparison: replies on screen are the recorded ones; decision matches.
  await page.getByRole('tab', { name: 'Response Comparison' }).click(); await page.waitForTimeout(800);
  const trace = await get(page, `/api/emotion/trace/${lastTurn}`);
  const shown = await page.locator('#mood-panel').innerText();
  check('unsteered reply shown equals the record', shown.includes(trace.effect.unsteered_reply.slice(0, 60)));
  check('steered reply shown equals the record (when one exists)', !trace.effect.steered_reply || shown.includes(trace.effect.steered_reply.slice(0, 60)));
  check('user message shown', shown.includes(MESSAGES[MESSAGES.length - 1].slice(0, 40)));
  check('planned and applied strength shown', shown.includes(`planned α ${trace.decision.planned_alpha.toFixed(2)}`) && shown.includes(`applied α ${trace.decision.applied_alpha.toFixed(2)}`));

  // Neuron Explorer: pick a turn with a steered candidate, load the full table, compare the first row with the record.
  await page.getByRole('tab', { name: 'Neuron Explorer' }).click(); await page.waitForTimeout(800);
  const steeredIndex = view.trajectory.responses.findIndex((row) => row.compared);
  if (steeredIndex >= 0) {
    await page.locator('.turn-picker button').nth(steeredIndex).click(); await page.waitForTimeout(600);
    const turnId = view.trajectory.responses[steeredIndex].turn_id;
    const record = await get(page, `/api/emotion/neurons/${turnId}`);
    await page.getByRole('button', { name: /Load the full neuron table/ }).click();
    await page.waitForSelector('.pager', { timeout: 20000 });
    const firstRow = await page.locator('article:has(.pager) tbody tr').first().locator('th,td').allTextContents();
    const top = record.neurons[0];
    check('full neuron table: first row is the record\'s top neuron', Number(firstRow[1]) === top.layer && Number(firstRow[2]) === top.neuron, firstRow.slice(0, 7).join(' | '));
    check('full neuron table: signed shift equals the record', Math.abs(Number(firstRow[6].replace('+', '')) - top.applied_shift) < 0.0006, `${firstRow[6]} vs ${top.applied_shift}`);
    check('full neuron table: modified = original + shift', Math.abs((top.steered_activation - top.unsteered_activation) - top.measured_change) < 0.001);
    const pagerText = await page.locator('.pager span').first().innerText();
    check('pager counts every neuron of the record', pagerText.includes(record.neurons.length.toLocaleString()), pagerText);
    const heatRows = await page.locator('.heat.neurons tbody tr').count();
    check('heat map has one row per steered layer', heatRows === new Set(record.neurons.map((n) => n.layer)).size, heatRows);
    await page.locator('.heat.neurons tbody tr th button').first().click(); await page.waitForTimeout(400);
    const filtered = await page.locator('.filter-row span.muted').first().innerText();
    check('clicking a heat-map layer filters the neuron table', /neurons match/.test(filtered) && !filtered.startsWith(record.neurons.length.toLocaleString()), filtered);
    await page.screenshot({ path: `${OUT}/neuron_explorer_table.png`, fullPage: true });
  } else check('a turn with a steered candidate exists for the neuron table', false);
  const panelName = Object.keys(research.panels)[0];
  const panelView = await get(page, `/api/emotion/panel/${panelName}`);
  check('panel browser endpoint lists every panel neuron', panelView.neurons.length === research.panels[panelName], `${panelName}: ${panelView.neurons.length}`);

  // Causal Ablations: verdict and inspector numbers equal the saved report.
  await page.getByRole('tab', { name: 'Causal Ablations' }).click(); await page.waitForTimeout(800);
  const causalText = await page.locator('#mood-panel').innerText();
  if (research.validation) {
    await page.locator('.filter-row select').first().selectOption('pooled'); await page.waitForTimeout(500);
    const pooled = research.validation.pooled;
    const reference = pooled.conditions.find((row) => row.family === 'dose' && row.alpha === research.validation.reference_alpha);
    const text = await page.locator('#mood-panel').innerText();
    const sign = (v) => (v > 0 ? '+' : '') + v.toFixed(2);
    check('causal: reference-strength emotion change equals the report', text.includes(sign(reference.emotion.mean)), sign(reference.emotion.mean));
    check('causal: individual neurons are marked not tested', /Individual neurons[\s\S]{0,40}Not tested/.test(text));
    check('causal: usable-strength finding shown', text.includes('Can it be used without harming replies'));
    await page.locator('article:has(h2:text("Inspect one intervention")) select').selectOption('band_only_mid'); await page.waitForTimeout(400);
    const band = pooled.conditions.find((row) => row.name === 'band_only_mid');
    const inspector = await page.locator('article:has(h2:text("Inspect one intervention"))').innerText();
    check('causal inspector: layer-band values equal the report', inspector.includes((band.emotion.mean > 0 ? '+' : '') + band.emotion.mean.toFixed(3)) && inspector.includes(String(band.neurons.toLocaleString())), `${band.emotion.mean} / ${band.neurons}`);
  } else check('causal: empty state names the missing battery', causalText.includes('has not been run'));

  // Model Evaluation: table values equal the saved results; missing data is labelled.
  await page.getByRole('tab', { name: 'Model Evaluation' }).click(); await page.waitForTimeout(800);
  const modelText = await page.locator('#mood-panel').innerText();
  for (const model of models.models) {
    if (model.battery) {
      const dose = model.battery.doses[0];
      check(`model evaluation: ${model.label} weakest-strength gain equals the report`, modelText.includes((dose.emotion.mean > 0 ? '+' : '') + dose.emotion.mean.toFixed(2)), dose.emotion.mean);
    } else check(`model evaluation: ${model.label} is labelled ${model.battery_status}`, modelText.toLowerCase().includes(model.battery_status.slice(0, 11).toLowerCase()));
  }
  check('model evaluation: missing Hinglish battery data is labelled not available', modelText.includes('Not available'));
  await page.locator('.filter-row select').nth(1).selectOption('hinglish'); await page.waitForTimeout(400);
  check('language filter keeps only Hinglish rows in the detector table', !(await page.locator('article:has(h2:text("emotion detector")) tbody').innerText()).includes('English'));
  report.sources = [...new Set(Object.values(report.tabs).flatMap((tab) => tab.badges || []))];
  check('evidence labels present', ['MEASURED', 'CLASSIFIER-ESTIMATED', 'LLM-JUDGED', 'USER-REPORTED', 'NOT AVAILABLE'].every((word) => report.sources.some((text) => text.includes(word))), report.sources.join('; '));

  // Self-rating updates the record.
  await page.getByRole('tab', { name: 'Emotion Timeline' }).click(); await page.waitForTimeout(600);
  await page.getByRole('radiogroup', { name: 'My mood at turn 1' }).getByRole('radio', { name: '2' }).click(); await page.waitForTimeout(1200);
  const after = await get(page, `/api/emotion/conversation/${conversation}`);
  check('a self-rating is stored and returned by the backend', after.trajectory.turns[0].self_rating === 2 && after.trajectory.self_report.n === 1);

  // Narrow screen and dark theme.
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('tab', { name: 'Overview' }).click(); await page.waitForTimeout(800);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  check('no horizontal page overflow at phone width', overflow <= 1, overflow);
  await page.screenshot({ path: `${OUT}/overview_phone.png`, fullPage: true });
  await page.setViewportSize({ width: 1360, height: 900 });
  await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${OUT}/overview_dark.png`, fullPage: true });
  check('no browser errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  report.errors = errors; report.conversation = conversation;
  fs.writeFileSync(`${OUT}/dashboard_check.json`, JSON.stringify(report, null, 1));
  const failed = report.checks.filter((item) => !item.ok);
  console.log(`\n${report.checks.length - failed.length} of ${report.checks.length} checks passed`);
  await browser.close();
  process.exit(failed.length ? 1 : 0);
})().catch((e) => { console.error('FAILED', e.message); process.exit(1); });
