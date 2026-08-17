import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";

const arcColors = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"];
const GROUP_EMOJI = { Paint: "🎨", Scissors: "✂️", Mirror: "🪞", Magnet: "🧲", Magic: "✨" };
const JOURNAL_KEY = "neurogolf-doors-v1";

function loadJournal() {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(JOURNAL_KEY) || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function saveJournal(journal) {
  window.localStorage.setItem(JOURNAL_KEY, JSON.stringify(journal));
}

function sameMove(a, b) {
  return a && b && a.op === b.op && JSON.stringify(a.params || {}) === JSON.stringify(b.params || {});
}

function moveLabel(move, library) {
  const door = library.find((item) => item.op === move.op);
  const base = door ? `${door.emoji} ${door.label}` : move.op;
  const params = Object.entries(move.params || {}).map(([, value]) => value).join(",");
  return params ? `${base} (${params})` : base;
}

const LIGHT_COLORS = new Set([3, 4, 5, 7, 8]); // digits need dark text on these

function DoorGrid({ values, target, onCellClick, size = "md", title, showNumbers = false }) {
  const rows = values?.length || 0;
  const cols = values?.[0]?.length || 0;
  if (!rows || !cols) return <div className="doorGridEmpty">?</div>;
  const shapeOk = target && target.length === rows && target[0]?.length === cols;
  const widthPx = size === "md" ? 170 : size === "sm" ? 110 : 44;
  const digitSize = Math.floor((widthPx / cols) * 0.62);
  const digits = showNumbers && size !== "xs" && digitSize >= 3;
  return (
    <div className="doorGridBlock">
      {title && <div className="doorGridTitle">{title} <span>{rows}x{cols}</span></div>}
      <div
        className={`doorGrid ${size}`}
        style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, aspectRatio: `${cols} / ${rows}` }}
      >
        {values.flatMap((row, r) =>
          row.map((value, c) => {
            const mismatch = shapeOk && target[r][c] !== value;
            const match = shapeOk && target[r][c] === value;
            return (
              <div
                key={`${r}-${c}`}
                className={`doorCell ${mismatch ? "bad" : match ? "good" : ""}`}
                style={{ backgroundColor: arcColors[value] || "#1e293b", cursor: onCellClick ? "crosshair" : "default" }}
                onClick={onCellClick ? () => onCellClick(value, r, c) : undefined}
              >
                {digits && (
                  <span className={`cellDigit ${LIGHT_COLORS.has(value) ? "dark" : ""}`}
                    style={{ fontSize: `${digitSize}px` }}>
                    {value}
                  </span>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

function Meter({ value, win }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div className="doorMeter">
      <div className={`doorMeterFill ${win ? "win" : ""}`} style={{ width: `${pct}%` }} />
      <span>{win ? "SOLVED!" : `${pct}% match`}</span>
    </div>
  );
}

function ParamPicker({ door, paramSpec, onPick, onCancel }) {
  if (paramSpec.type === "color") {
    return (
      <div className="paramPicker">
        <strong>{door.emoji} {door.label}: pick {paramSpec.name} color</strong>
        <em>(click a swatch, or click a cell on any picture)</em>
        <div className="paletteRow">
          {arcColors.map((color, index) => (
            <button key={color} className="swatch" style={{ background: color }} onClick={() => onPick(index)}>
              <span>{index}</span>
            </button>
          ))}
        </div>
        <button className="btn compact" onClick={onCancel}>Cancel</button>
      </div>
    );
  }
  if (paramSpec.type === "direction") {
    return (
      <div className="paramPicker">
        <strong>{door.emoji} {door.label}: which way?</strong>
        <div className="paletteRow">
          {[["up", "⬆️"], ["down", "⬇️"], ["left", "⬅️"], ["right", "➡️"]].map(([dir, icon]) => (
            <button key={dir} className="btn" onClick={() => onPick(dir)}>{icon} {dir}</button>
          ))}
        </div>
        <button className="btn compact" onClick={onCancel}>Cancel</button>
      </div>
    );
  }
  if (paramSpec.type === "turns") {
    return (
      <div className="paramPicker">
        <strong>{door.emoji} {door.label}: how far?</strong>
        <div className="paletteRow">
          {[[1, "90°"], [2, "180°"], [3, "270°"]].map(([turns, label]) => (
            <button key={turns} className="btn" onClick={() => onPick(turns)}>{label}</button>
          ))}
        </div>
        <button className="btn compact" onClick={onCancel}>Cancel</button>
      </div>
    );
  }
  return (
    <div className="paramPicker">
      <strong>{door.emoji} {door.label}: {paramSpec.name}?</strong>
      <div className="paletteRow">
        {[1, 2, 3, 4, 5].map((n) => <button key={n} className="btn" onClick={() => onPick(n)}>{n}</button>)}
      </div>
      <button className="btn compact" onClick={onCancel}>Cancel</button>
    </div>
  );
}

export default function DoorsView({ taskId, taskNumber, task, taskLoadState, setTaskNumber, onBack, baseline }) {
  const [library, setLibrary] = useState([]);
  const [campaign, setCampaign] = useState({ tutorials: [], leads: [], queue: [] });
  const [path, setPath] = useState([]);
  const [thumbs, setThumbs] = useState([]);
  const [board, setBoard] = useState(null);
  const [pending, setPending] = useState(null);
  const [ruleText, setRuleText] = useState("");
  const [exportResult, setExportResult] = useState(null);
  const [bestRule, setBestRule] = useState(null);
  const [status, setStatus] = useState("Pick a door!");
  const [busy, setBusy] = useState(false);
  const [wizard, setWizard] = useState(null);
  const [wizardBusy, setWizardBusy] = useState(false);
  const [showNumbers, setShowNumbers] = useState(() => window.localStorage.getItem("neurogolf-doors-numbers") === "1");

  const toggleNumbers = () => {
    setShowNumbers((current) => {
      window.localStorage.setItem("neurogolf-doors-numbers", current ? "0" : "1");
      return !current;
    });
  };
  const [studyMode, setStudyMode] = useState(() => window.localStorage.getItem("neurogolf-doors-study") !== "0");

  const toggleStudy = () => {
    setStudyMode((current) => {
      window.localStorage.setItem("neurogolf-doors-study", current ? "0" : "1");
      return !current;
    });
  };
  const journalRef = useRef(loadJournal());
  const examples = task?.train || [];

  useEffect(() => {
    axios.get("/api/door/library").then(({ data }) => setLibrary(data.doors || [])).catch(() => setLibrary([]));
    axios.get("/api/door/campaign").then(({ data }) => setCampaign(data)).catch(() => {});
  }, []);

  useEffect(() => {
    const saved = journalRef.current[taskId];
    setPath(saved?.path || []);
    setThumbs(saved?.thumbs || []);
    setRuleText(saved?.ruleText || "");
    setExportResult(null);
    setPending(null);
    setBoard(null);
    setWizard(null);
    setBestRule(null);
    setStatus("Pick a door!");
  }, [taskId]);

  useEffect(() => {
    if (taskLoadState !== "loaded") return;
    let cancelled = false;
    setBusy(true);
    axios.post("/api/door/step", { taskId, path })
      .then(({ data }) => {
        if (cancelled) return;
        setBoard(data);
        setThumbs((current) => {
          const next = current.slice(0, path.length);
          if (path.length > 0) next[path.length - 1] = data.examples?.[0]?.current || null;
          return next;
        });
        if (data.win) setStatus("🎉 All train examples match! Check + export below.");
      })
      .catch((error) => {
        if (cancelled) return;
        const reason = error.response?.data?.reason || error.message;
        setStatus(`That door won't open: ${reason}`);
        setPath((current) => current.slice(0, -1));
      })
      .finally(() => !cancelled && setBusy(false));
    return () => { cancelled = true; };
  }, [taskId, path, taskLoadState]);

  useEffect(() => {
    journalRef.current[taskId] = { path, thumbs, ruleText, solved: Boolean(board?.win) };
    saveJournal(journalRef.current);
  }, [taskId, path, thumbs, ruleText, board]);

  const ghost = useMemo(() => {
    const entry = [...(campaign.tutorials || []), ...(campaign.leads || [])].find((item) => item.taskId === taskId);
    if (!entry?.ghostPath) return null;
    const isPrefix = path.every((step, index) => sameMove(step, entry.ghostPath[index]));
    return isPrefix && path.length < entry.ghostPath.length ? entry.ghostPath[path.length] : null;
  }, [campaign, taskId, path]);

  const lookahead = board?.lookahead || [];
  const glowOps = new Set(lookahead.map((move) => move.op));

  const applyMove = useCallback((move) => {
    setExportResult(null);
    setPath((current) => [...current, { op: move.op, params: move.params || {} }]);
    setStatus("…");
  }, []);

  const clickDoor = (door) => {
    if (busy) return;
    if (!door.params.length) {
      applyMove({ op: door.op, params: {} });
      return;
    }
    setPending({ door, values: {} });
  };

  const pendingSpec = pending ? pending.door.params[Object.keys(pending.values).length] : null;

  const pickParam = (value) => {
    if (!pending || !pendingSpec) return;
    const values = { ...pending.values, [pendingSpec.name]: value };
    if (Object.keys(values).length === pending.door.params.length) {
      setPending(null);
      applyMove({ op: pending.door.op, params: values });
    } else {
      setPending({ ...pending, values });
    }
  };

  const onCellClick = (value) => {
    if (pending && pendingSpec?.type === "color") pickParam(value);
  };

  const rewindTo = (index) => {
    setExportResult(null);
    setPath((current) => current.slice(0, index));
    setStatus(index === 0 ? "Back to the start." : `Rewound to step ${index}.`);
  };

  const exportPath = async () => {
    setBusy(true);
    setStatus("Checking everything and exporting…");
    try {
      const { data } = await axios.post("/api/door/export", { taskId, path, ruleHypothesis: ruleText });
      setExportResult(data);
      setStatus(data.route === "auto_compile"
        ? `⭐⭐⭐ Compiled + officially scored! Saved ${data.artifact}`
        : `⭐ Recipe saved for the agents: ${data.recipe}`);
    } catch (error) {
      const reason = error.response?.data?.reason || error.message;
      setExportResult(null);
      setStatus(`Export failed: ${reason}`);
    } finally {
      setBusy(false);
    }
  };

  const saveRuleNote = async () => {
    try {
      const { data } = await axios.post("/api/door/rule-note", { taskId, path, ruleHypothesis: ruleText });
      setStatus(`📝 Rule note saved for the agents: ${data.recipe}`);
    } catch (error) {
      setStatus(`Rule note failed: ${error.response?.data?.reason || error.message}`);
    }
  };

  const startFresh = () => {
    setPath([]);
    setThumbs([]);
    setRuleText("");
    setExportResult(null);
    setWizard(null);
    setPending(null);
    delete journalRef.current[taskId];
    saveJournal(journalRef.current);
    setStatus("🧼 Fresh start — blank slate, blank notebook.");
  };

  const revealBestRule = async () => {
    try {
      const { data } = await axios.get(`/api/task-rule/${taskId}`);
      setBestRule(data);
      setStatus(data.known ? "📖 Best known rule revealed." : "No decoded rule yet — you're the explorer!");
    } catch (error) {
      setBestRule(null);
      setStatus(`Rule lookup failed: ${error.response?.data?.reason || error.message}`);
    }
  };

  const tryPath = (steps) => {
    if (!steps?.length) return;
    setExportResult(null);
    setThumbs([]);
    setPath(steps.map((step) => ({ op: step.op, params: step.params || {} })));
    setStatus("Replaying the best-known path…");
  };

  const askWizard = async (backend) => {
    setWizardBusy(true);
    setWizard(null);
    setStatus(`🧙 The ${backend} wizard is thinking… (can take a few minutes)`);
    try {
      const { data } = await axios.post("/api/door/wizard", { taskId, path, ruleHypothesis: ruleText, backend });
      setWizard(data);
      setStatus(data.replay?.win
        ? "🧙 The wizard found a WINNING path — try it!"
        : data.doorPath?.length
          ? `🧙 Wizard suggests a path (${Math.round((data.replay?.distance || 0) * 100)}% match)`
          : "🧙 The wizard answered with an algorithm note (no door path)");
    } catch (error) {
      setStatus(`Wizard failed: ${error.response?.data?.reason || error.message}`);
    } finally {
      setWizardBusy(false);
    }
  };

  const tryWizardPath = () => {
    if (!wizard?.doorPath?.length) return;
    setExportResult(null);
    setThumbs([]);
    setPath(wizard.doorPath.map((step) => ({ op: step.op, params: step.params || {} })));
    setStatus("Trying the wizard's path…");
  };

  const groups = useMemo(() => {
    const byGroup = new Map();
    for (const door of library) {
      if (!byGroup.has(door.group)) byGroup.set(door.group, []);
      byGroup.get(door.group).push(door);
    }
    return Array.from(byGroup.entries());
  }, [library]);

  const score = exportResult?.score;

  return (
    <div className="doorsShell">
      <aside className="doorsLeft">
        <button className="btn full" onClick={onBack}>← Lab mode</button>
        <h2>TUTORIALS</h2>
        {(campaign.tutorials || []).map((item) => (
          <button key={item.taskId} className={`btn full ${item.taskId === taskId ? "activeMode" : ""}`}
            onClick={() => setTaskNumber(Number(item.taskId.replace("task", "")))}>
            {item.label}
          </button>
        ))}
        <h2>WARM LEADS</h2>
        {(campaign.leads || []).map((item) => (
          <button key={item.taskId} className={`btn full ${item.taskId === taskId ? "activeMode" : ""}`}
            onClick={() => setTaskNumber(Number(item.taskId.replace("task", "")))}>
            🔥 {item.taskId} — {item.label}
          </button>
        ))}
        <h2>CAMPAIGN</h2>
        <div className="campaignList">
          {(campaign.queue || []).slice(0, 20).map((item) => (
            <button key={item.taskId} className={item.taskId === taskId ? "active" : ""}
              onClick={() => setTaskNumber(Number(item.taskId.replace("task", "")))}>
              {item.taskId.replace("task", "")}
            </button>
          ))}
        </div>
        {baseline && Number.isFinite(baseline.currentPoints) && (
          <p className="muted">Live best: cost {baseline.currentCost} / {baseline.currentPoints.toFixed(2)} pts</p>
        )}
      </aside>

      <main className="doorsMain">
        <header className="doorsHeader">
          <div>
            <strong>🚪 Doors Maze — {taskId}</strong>
            <span className="muted"> {examples.length} examples{studyMode ? " — study the pairs, then say the rule" : ", every door opens on ALL of them"}</span>
          </div>
          <div className="doorsNav">
            <button className={`btn compact ${studyMode ? "activeMode" : ""}`} onClick={toggleStudy}
              title="Study mode: just the example pairs + rule writing, no doors game">👁 Just look</button>
            <button className={`btn compact ${showNumbers ? "activeMode" : ""}`} onClick={toggleNumbers}
              title="Show the number behind every color (colors ARE digits 0-9; some rules are math)">123</button>
            <button className="btn compact" onClick={() => setTaskNumber(taskNumber - 1)}>◀</button>
            <button className="btn compact" onClick={() => setTaskNumber(taskNumber + 1)}>▶</button>
          </div>
        </header>

        {!studyMode && <Meter value={board?.distance} win={Boolean(board?.win)} />}

        <section className={studyMode ? "doorsBoards study" : "doorsBoards"}>
          {examples.map((example, index) => (
            <div key={index} className="doorsExample">
              <DoorGrid values={example.input} title="IN" size={studyMode ? "md" : "sm"} onCellClick={onCellClick} showNumbers={showNumbers} />
              <span className="doorsArrow">→</span>
              {studyMode ? (
                <DoorGrid values={example.output} title="OUT" size="md" showNumbers={showNumbers} />
              ) : (
                <>
                  <DoorGrid
                    values={board?.examples?.[index]?.current || example.input}
                    target={example.output}
                    title="NOW"
                    size="md"
                    onCellClick={onCellClick}
                    showNumbers={showNumbers}
                  />
                  <span className="doorsArrow">vs</span>
                  <DoorGrid values={example.output} title="GOAL" size="sm" showNumbers={showNumbers} />
                </>
              )}
            </div>
          ))}
        </section>

        {!studyMode && path.length > 0 && (
          <section className="filmstrip">
            <button className="filmstripStep" onClick={() => rewindTo(0)} title="Back to start">⟲ start</button>
            {path.map((step, index) => (
              <button key={index} className="filmstripStep" onClick={() => rewindTo(index + 1)}
                title={`Rewind to after step ${index + 1}`}>
                <span>{moveLabel(step, library)}</span>
                {thumbs[index] && <DoorGrid values={thumbs[index]} size="xs" />}
              </button>
            ))}
            <button className="btn compact" onClick={() => rewindTo(path.length - 1)} disabled={busy}>Undo</button>
          </section>
        )}

        {!studyMode && (pending && pendingSpec ? (
          <ParamPicker door={pending.door} paramSpec={pendingSpec} onPick={pickParam} onCancel={() => setPending(null)} />
        ) : (
          <>
            {lookahead.length > 0 && !board?.win && (
              <div className="bestMoves">
                <span>✨ Glowing doors:</span>
                {lookahead.map((move, index) => (
                  <button key={index} className="chip" onClick={() => applyMove(move)} disabled={busy}>
                    {move.label} +{Math.round(move.gain * 100)}%
                  </button>
                ))}
              </div>
            )}
            <section className="doorRack">
              {groups.map(([group, doors]) => (
                <div key={group} className="doorGroup">
                  <h3>{GROUP_EMOJI[group] || ""} {group}</h3>
                  <div className="doorGroupDoors">
                    {doors.map((door) => {
                      const isGhost = ghost && ghost.op === door.op;
                      return (
                        <button key={door.op}
                          className={`doorBtn ${glowOps.has(door.op) ? "glow" : ""} ${isGhost ? "ghostGlow" : ""}`}
                          onClick={() => clickDoor(door)} disabled={busy} title={door.blurb}>
                          <span className="doorEmoji">{door.emoji}</span>
                          <span>{door.label}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </section>
          </>
        ))}

        <div className="doorsStatus">{status}</div>
      </main>

      <aside className="doorsRight">
        <button className="btn full" onClick={startFresh} disabled={busy}
          title="Clear the door path, your rule notes, and the saved journal for this task">
          🧼 Start from scratch
        </button>
        {!studyMode && board?.win && (
          <div className="winPanel">
            <h2>🎉 SOLVED</h2>
            {board.sets && Object.entries(board.sets).map(([name, counts]) => (
              <div key={name} className={counts.fail ? "setBad" : "setGood"}>
                {name}: {counts.pass}/{counts.pass + counts.fail}
              </div>
            ))}
            <button className="btn primary full" onClick={exportPath} disabled={busy}>Export solution</button>
            {exportResult && (
              <div className="exportResult">
                <div>Route: <strong>{exportResult.route}</strong></div>
                {score?.scored && (
                  <>
                    <div>Official: {score.memory} mem + {score.params} params = cost {score.cost}</div>
                    <div>Points: {score.points.toFixed(3)}
                      {Number.isFinite(exportResult.deltaPoints) ? ` (delta ${exportResult.deltaPoints >= 0 ? "+" : ""}${exportResult.deltaPoints.toFixed(3)})` : ""}
                    </div>
                  </>
                )}
                {exportResult.autoCompileError && (
                  <div className="muted">Auto-compile didn't fit (geometry varies) — the recipe is saved for the agents to golf.</div>
                )}
                <div className="muted">{exportResult.artifact || exportResult.recipe}</div>
              </div>
            )}
          </div>
        )}

        <div className="rulePanel">
          <h2>SAY THE RULE</h2>
          <textarea value={ruleText} onChange={(event) => setRuleText(event.target.value)}
            placeholder="What's the rule? e.g. 'copy the pattern into every colored cell of a 9x9'. Words count even without doors!" />
          <div className="ruleButtons">
            <button className="btn" onClick={saveRuleNote} disabled={!ruleText.trim()}>📝 Save rule note</button>
            <button className="btn" onClick={() => askWizard("claude")} disabled={wizardBusy}>🧙 Ask Claude</button>
            <button className="btn" onClick={() => askWizard("codex")} disabled={wizardBusy}>🧙 Ask Codex</button>
          </div>
          {wizardBusy && <div className="muted">The wizard is reading the scrolls…</div>}
          {wizard && (
            <div className="wizardResult">
              {wizard.ruleHypothesis && <div><strong>Wizard:</strong> {wizard.ruleHypothesis}</div>}
              {wizard.doorPath?.length > 0 && (
                <>
                  <div className="muted">{wizard.doorPath.map((step) => moveLabel(step, library)).join(" → ")}</div>
                  <div>
                    {wizard.replay?.win ? "✅ wins all train examples!" :
                      wizard.replay?.error ? `⚠️ ${wizard.replay.error}` :
                      `gets to ${Math.round((wizard.replay?.distance || 0) * 100)}%`}
                  </div>
                  <button className="btn full" onClick={tryWizardPath}>Try wizard path</button>
                </>
              )}
              {wizard.notes && <div className="muted">{wizard.notes}</div>}
            </div>
          )}
        </div>

        <div className="bestRulePanel">
          <h2>BEST KNOWN RULE</h2>
          {!bestRule ? (
            <button className="btn full" onClick={revealBestRule}>📖 Reveal (spoiler!)</button>
          ) : (
            <>
              {bestRule.known ? (
                <pre className="bestRuleText">{bestRule.rule}</pre>
              ) : (
                <div className="muted">No decoded rule for {taskId} yet — NOBODY knows it. Discover it and 📝 save yours!</div>
              )}
              {bestRule.ghostPath?.length > 0 && (
                <div className="ghostExplain">
                  <strong>As doors:</strong>
                  {bestRule.ghostPath.map((step, index) => {
                    const door = library.find((item) => item.op === step.op);
                    const params = Object.entries(step.params || {}).map(([key, value]) => `${key}: ${value}`).join(", ");
                    return (
                      <div key={index}>
                        {index + 1}. {door ? `${door.emoji} ${door.label}` : step.op}{params ? ` (${params})` : ""}
                        {door?.blurb ? <span className="muted"> — {door.blurb}</span> : null}
                      </div>
                    );
                  })}
                </div>
              )}
              {!studyMode && bestRule.ghostPath?.length > 0 && (
                <button className="btn full" onClick={() => tryPath(bestRule.ghostPath)} disabled={busy}>
                  ▶ Watch the known door path ({bestRule.ghostPath.length} {bestRule.ghostPath.length === 1 ? "door" : "doors"})
                </button>
              )}
              {!studyMode && (bestRule.recipes || []).filter((item) => item.path?.length).slice(0, 3).map((item) => (
                <button key={item.file} className="btn full" onClick={() => tryPath(item.path)} disabled={busy}
                  title={item.ruleHypothesis || item.file}>
                  ▶ Replay saved {item.exportRoute === "hypothesis_only" ? "note" : "recipe"} ({item.path.length} doors)
                </button>
              ))}
              <button className="btn compact" onClick={() => setBestRule(null)}>Hide</button>
            </>
          )}
        </div>

        {studyMode ? (
          <div className="howTo">
            <h2>STUDY MODE</h2>
            <p>Look at each <strong>IN → OUT</strong> pair. What one story explains ALL of them?</p>
            <p>Write it in <strong>SAY THE RULE</strong> and 📝 save it — your words go straight to the agents.</p>
            <p>Stuck or curious? 📖 reveal the best known rule, or 🧙 ask the wizard.</p>
            <p>Want to act the rule out move by move? Flip off 👁 Just look up top.</p>
          </div>
        ) : !board?.win && (
          <div className="howTo">
            <h2>HOW TO PLAY</h2>
            <p>Make every <strong>NOW</strong> picture match its <strong>GOAL</strong> by opening doors.</p>
            <p>One door changes <em>all</em> the examples at once — that's what makes it a rule!</p>
            <p>✨ Glowing doors look promising. Click a step in the filmstrip to go back.</p>
            <p>Picking a color? You can click any cell on the pictures.</p>
            <p>Know the rule but not the doors? Write it down and 🧙 ask the wizard — or just save the note; it goes straight to the agents.</p>
          </div>
        )}
      </aside>
    </div>
  );
}
