# PromptManager — CLI quick wins

Status: plan QW1–QW5 wykonany — QW1 dostarczony `cd0a3a5`, Quality Gates PASS; QW2–QW5 ukończone lokalnie, commit/push zatwierdzony i w toku. Wynik remote/CI jeszcze do potwierdzenia.
Owner: Wojtek / Prompt Manager Team
Baseline: `master@29e6bdbbccdf06dfcb64bf43f38eade508d6942c`, czyste drzewo przed utworzeniem planu.
Authority: `docs/product-ssot.md`; ten dokument jest trackerem wykonania, nie zmianą kierunku produktu.

## Cel i stan potwierdzony

Małe naprawy istniejących kontraktów CLI pod kontrolowane użycie przez lokalnego agenta. Nie kwalifikować całego CLI jako agent-ready po ich wykonaniu.

Audyt baseline: 369 testów CLI PASS; 197 prób procesowych (w tym 138 help), Ruff/format i lock PASS. Jawny `pyright cli`: 123 odziedziczone błędy w `cli/commands.py`. Pełnego suite/coverage w tym audycie nie powtarzano. Próby provider-free wykazały m.in. output-path failure po execution, tekstowy receipt przy chain JSON/file, exit 0 mimo failed run, offline warnings w JSON oraz różnicę w early dispatch prompt-edit.

## Zasady zgody i wykonania

- **Pytać użytkownika przed rozpoczęciem każdego quick win, również QW1.** Zgoda na zapis planu i rozpoczęcie prac organizacyjnych nie pomija tej bramki. Zgoda obejmuje tylko wskazany QW; po jego zakończeniu zatrzymać się i zapytać o kolejny. Ogólne „proceduj” przy pytaniu dotyczącym QW oznacza ten jeden QW, nie cały plan.
- Po zgodzie zmienić status QW na aktywny; po każdym istotnym kroku dopisać fakty do tego samego dziennika. Nie oznaczać completed przed GREEN, wymaganymi bramkami, docs i readback.
- Kod, testy i Git tylko w `/home/voytas/projects/PromptManager`; scratch/probes poza repo. Bez modyfikacji Windows checkout i danych użytkownika.
- Zgoda „Commit i push a potem qw2” obejmuje staging/commit/push zamkniętego QW1 i rozpoczęcie QW2 po dostarczeniu QW1. Nie obejmuje commit/push QW2, QW3–QW5, zmian zależności/lock, CI/config/security policy ani modeli/providerów live.
- Zachować publiczne kontrakty poza zatwierdzonym QW. Bez globalnego wyciszania logowania i szerokiej ekstrakcji `commands.py`.
- Przy konieczności poszerzenia zakresu lub zmiany polityki zatrzymać się i uzyskać zgodę. Preferować najmniejszy odwracalny wariant.

## QW1 — preflight output przed chain execution

Status: dostarczony `cd0a3a5` na origin/master — trzy SHA zgodne, Quality Gates PASS.
Cel: oczywisty lokalny błąd wyjścia nie może uruchamiać chain/modelu; późniejsza awaria zapisu nie może kończyć się tracebackiem.
Zakres przewidywany: `cli/commands.py`, ewentualnie mały helper w istniejącym `cli/utils.py`, `tests/test_prompt_chain_cli.py` lub osobny wąski test procesowy, `docs/README-DEV.md`, `docs/CHANGELOG.md`, ten tracker. Bez zmiany parsera/flag, providerów, run-status/exit sukcesu ani formatu udanego receipt (QW2/QW3).

Kroki:
1. Odczytać aktywne helpery/politykę output oraz sąsiednie testy. Zachować dotychczasowe nadpisywanie zwykłego pliku; nowa odmowa nadpisania wymaga osobnej decyzji. Nie nazywać samego preflight gwarancją późniejszego zapisu.
2. RED przez oba publiczne entrypointy z injected fake managerem: parent będący plikiem, target będący katalogiem oraz błąd przygotowania wyjścia => zero run calls, pusty stdout, nonzero exit, sanitarny błąd bez ścieżki/inputu/tracebacku. Dodać sukces z nowym parentem i kontrolę dotychczasowego nadpisania.
3. Przygotować wyjście przed run (również text/file), zachowując brak przygotowania pliku bez `--output-file`. Nie kasować katalogów/plików użytkownika przy błędzie. Późniejszą awarię zapisu przetłumaczyć na bounded error; jedno wykonanie może już nastąpić i nie wolno uruchamiać go ponownie.
4. GREEN oraz process smoke: zero calls przy odmowie preflight; exactly one przy późniejszej awarii i normalnym sukcesie; odczyt rzeczywistego artefaktu. Kwestie races/writability kwalifikować uczciwie.
5. Bramki, docs/changelog i checkpoint. Zatrzymać się przed QW2.

Done: niepoprawny output odrzucony przed execution w obu entrypointach; zachowany poprawny zapis/nadpisanie/text, późniejszy błąd bez tracebacku; testy i wymagane bramki potwierdzone. Preflight nie usuwa kosztów bootstrapu managera ani races po sprawdzeniu.

## QW2 — receipt i runtime-error dla chain JSON

Status: ukończony lokalnie — implementacja, testy/bramki/docs i niezależny review PASS; nie commitowano/pushowano.
Zakres: handler chain i renderer receipt, odpowiednie testy procesowe, developer guide/changelog/tracker.
- `--json --output-file`: jeden JSON receipt `{command:"prompt-chain-run",artifact_path:PATH,run_status:STATUS}` po zapisie zamiast tekstowego `Saved ...`. Path zachowuje zapis argumentu jak dotychczas; nie dodawać misleading `ok:true` dla failed run. Status jak obecny handler `result.run_status or "success"`; pełny payload pliku i JSON stdout bez pliku bez zmian. Receipt nie zawiera inputu/outputu/name/provider metadata.
- Wyjątki chain w JSON: sanitarny error JSON stderr, pusty stdout, zachowane kody błędów. Nie echo surowego provider exception.
- Schemat błędu jak QW1 `{ok:false,command:"prompt-chain-run",error:{code,message}}`, exit 5. `PromptChainExecutionError` => `CHAIN_EXECUTION_FAILED`, `PromptChainError` => `CHAIN_RUN_FAILED`, unexpected `Exception` w samym runner call => `CHAIN_RUN_FAILED` tylko w JSON (text pozostawia dotychczasowy wyjątek). Nie obejmuje parsera/startupu, invalid input/selectors ani serializacji sukcesu całej rodziny.
- Zachować default text, payload JSON bez pliku i run-status semantics do QW3.
Done: sukces/wyjątek/file-mode z obu entrypointów, readback artefaktu, marker privacy i bramki. Parser/startup całej rodziny nie jest domyślnie zakresem QW2.

## QW3 — exit code a domain success

Status: ukończony lokalnie — kontrakt, docs, pełne bramki i niezależny review PASS; nie commitowano/pushowano.
Zakres: istniejące chain/benchmark handlers, testy sukces/partial/failure, docs/changelog/tracker.
- Przed kodem ustalić mapowanie success/partial/failed/no-runs do exit oraz zidentyfikować istniejące oczekiwania klientów/testów.
- Failed chain i benchmark all-errors nie powinny udawać sukcesu run. Wariant węższy: opt-in strict status, jeżeli zmiana domyślnego exit łamie potwierdzonego klienta.
- Zachować materiał diagnostyczny i zapis artefaktu nieudanego run tam, gdzie jest częścią istniejącego kontraktu. Nie dodawać benchmark JSON ani nowych retries w tym QW bez zgody.
Done: kontrakt zaakceptowany, deterministyczne run outcome fixtures i real-process exit/output parity, docs i bramki.

Kontrakt QW3 (zatwierdzony zakres, bez nowych flag):
- Chain: exit **0 tylko dla `run_status == "success"`**; `partial_success`, `failed`, `skipped` i status pusty/nieznany -> **5**. Nie wyliczać ponownie statusu z kroków w CLI. Pusty status prezentować jako `unknown`, nie domyślny sukces; pełny JSON zachowuje surowy backend `run_status`.
- Benchmark: exit **0 tylko gdy `runs` jest niepuste i każdy `run.error is None`**; mixed/all-errors/no-runs -> **5**. Pusty string błędu jest błędem, zgodnie z sentinel `None`, również w tekście (`ERROR`, nie `OK`). Brak oceniania jakości/treści odpowiedzi w QW3.
- Wyniki/receipt/zapis pliku powstają przed zwróceniem kodu domenowego. Returned partial/failed to nadal wynik stdout, nie exception envelope stderr; zachować materiał diagnostyczny. Output-path/runner errors nadal mają pierwszeństwo i obecny exit 5. Bez retry ani nowych efektów wykonania.
- Kompatybilność: nie znaleziono rzeczywistego klienta CLI w checkout; GUI wywołuje backend bezpośrednio i nie zależy od exit. Dotychczasowe asercje 0 dla partial/failed w testach są świadomie zmieniane w QW3. Zależności spoza repo pozostają do weryfikacji. Najsilniejsza alternatywa: opt-in strict status, jeśli ujawni się potwierdzony klient świadomie zależny od dawnego exit 0; teraz nie dodawać flag na spekulację.

## QW4 — oczekiwane offline warnings w wybranych JSON success

Status: ukończony lokalnie — JSON-only bootstrap, testy, pełne bramki, docs i niezależny review PASS; nie commitowano/pushowano.
Zakres: CommandSpec/bootstrap flags tylko dla `prompt-render --json` i `tag-list --json`, odpowiednie testy, docs/changelog/tracker.
- Użyć istniejącej kontroli `announce_offline_llm`; nie wyciszać błędów/globalnych loggerów ani zmieniać zwykłego text-mode.
- Próby ze scratch CWD i repozytoryjną przykładową konfiguracją logowania.
Done: jeden JSON stdout, pusty stderr na oczekiwanym offline sukcesie w obu entrypointach; default text bez regresji. Nie kwalifikuje bezpieczeństwa renderer input ani parser/startup/runtime errors całej rodziny.

Kontrakt: w istniejącym bootstrapie `main.py` przekazać `announce_offline_llm=False` wyłącznie gdy command jest `prompt-render` lub `tag-list` i `args.json` jest true. Nie ustawiać CommandSpec false dla wszystkich trybów, bo wyciszyłoby text. Zachować reason/available managera; factory wyłącza tylko własny expected warning i `notify`, nie logger level/handlers, inne warnings/errors ani wykonanie. Parser/startup/handler errors bez zmian. Nie twierdzić, że custom logging lub inne ostrzeżenia zawsze pozwalają na parseable JSON. Alternatywa szersza: CommandSpec false dla obu komend, tylko jeśli użytkownik zatwierdzi też zmianę text; obecnie poza zakresem.

## QW5 — early dispatch prompt-edit parity

Status: ukończony lokalnie — early module dispatch, pełne bramki, docs i niezależny review PASS; nie commitowano/pushowano.
Zakres: `main.py`, wąskie process/import-boundary tests, docs/changelog/tracker.
- W modułowym entrypoincie dispatch `prompt-edit` przed heavy runtime imports, analogicznie do installed launcher.
- Bez zmiany SQL/mutation/value/backup contract.
Done: oba entrypointy, forbidden import guard dla GUI/provider runtime, preview/error/apply na disposable SQLite i readback, brak niejawnego bootstrapu, bramki.

## Wspólna weryfikacja

- RED -> minimalne GREEN; fakes i fail-on-network, scrubbed provider env, izolowane HOME/CWD/config/TMPDIR/SQLite, absolute project interpreter.
- Sprawdzić realny installed wrapper i `python -m main`, oddzielnie exit/stdout/stderr/calls/artifact. Pomoc pozostaje success path.
- Ukierunkowane i sąsiednie testy; repo Ruff check i format --check (formatować tylko dotknięte pliki). Strict Pyright dotkniętych plików/testów; dla indebted commands.py porównać dokładną deltę do baseline, nie zmieniać CI/include/suppressions.
- Po zmianie produkcyjnej pełny provider-free pytest+coverage >=80%, lock check, diff check oraz scope inventory. Dziedziczony dług raportować osobno.
- Zmierzyć `git diff --numstat` plus nowe nieśledzone pliki; nie opisywać estymaty jako wyniku. Niezależny review proporcjonalny do ryzyka, ponowne bramki po poprawkach.

## Poza zakresem i wyzwalacze

Bezpieczny bounded renderer, validate-only bez ewaluacji, bootstrap-free preview, globalny CAS/idempotency, provider-independent budgets/deadlines, pełny CLI type cleanup i nowe API są odłożone. Niezaufane render/test oraz niekwalifikowane execution/reset nadal poza agent allowlistą. Quick wins nie usuwają tych blokad.

Alternatywa: wykonać najpierw QW4, jeśli potwierdzony konsument jest wyłącznie katalogowy i nie ma dostępu do chain/benchmark. Obecna kolejność pozostaje QW1 -> QW2 -> QW3 -> QW4 -> QW5, każdorazowo po osobnej zgodzie.

## Bieżące blokery / do weryfikacji / następny krok

- Bramka następnego zadania: QW1–QW5 wykonane, QW2–QW5 zamknięte lokalnie; commit/push zatwierdzony przez „Zatwierdzam” na pytanie o publikację. Nie dodawać kolejnego quick win bez nowej decyzji.
- Potwierdzone QW2: review PASS; końcowy focused 257 PASS, full 1401 PASS / 1 skipped, core coverage 82.12%; nowe executable statements 9/9 = 100%; Ruff/format/lock/diff PASS. Oba testy strict Pyright 0, commands.py odziedziczone 123 bez nowych fingerprintów.
- Potwierdzone QW3: review PASS; końcowy focused 422 PASS, full 1565 PASS / 1 skipped, core 82.12%; changed executable statements 10/10. Ruff/format, configured Pyright i strict trzy testy PASS; commands inherited123, fingerprint delta 0; lock/diff PASS.
- Potwierdzone QW4: focused 190 PASS, full 1611 PASS / 1 skipped, core 82.12%; strict main/test i configured Pyright PASS; commands inherited123 delta0, Ruff/format/lock/diff PASS, changed statements 2/2. Niezależny review PASS; sugestie logger/izolacja ustawień wdrożone i bramki powtórzone.
- Potwierdzone QW5: po sugestiach review pack 56 PASS, focused 194 PASS, full 1667 PASS / 1 skipped, core 82.12%, changed statements 3/3; Ruff/format, configured i strict main/test Pyright, lock/diff PASS; commands inherited123 delta0. Niezależny review PASS, bez security/logic findings, sugestie testowe wdrożone.
- Następny krok: commit/push QW2–QW5, trzy SHA i exact-SHA CI, następnie dokumentacyjny delivery checkpoint. Bez automatycznego poszerzania na odłożone prace autonomii/security.

## Dziennik wykonania

1. **Plan/intake:** potwierdzono clean master baseline, odczytano AGENTS, product SSOT i changelog. Zapisano pięć quick winów, osobną zgodę przed każdym (włącznie z pierwszym), granice provider/Git i kryteria weryfikacji. Brak zmian produkcyjnych na tym etapie.
2. **QW1 zgoda/discovery/RED:** użytkownik zatwierdził QW1 przez formularz; brak zgody na QW2–QW5. Zachowujemy tworzenie brakujących parentów i nadpisanie zwykłego pliku. Baseline `pyright cli`: 123 błędy. Dodano procesowy pack: oba entrypointy × JSON/text, odmowa przygotowania, nieprawidłowe parent/target, unwritable target, późniejsza awaria zapisu i kontrole sukcesu/no-file. RED: **20 failed / 12 passed**, oczekiwana utrata granicy preflight i traceback po zapisie; nie błąd harnessu. Evidence: `/home/voytas/.hermes/cache/scratch/promptmanager-qw1-tp2wlqu2/red.log`. Następny krok: minimalne GREEN.
3. **QW1 pierwsze GREEN:** procesowy pack **32 PASS**, sąsiedni pack **192 PASS**; pełne provider-free suite **1336 PASS / 1 skipped, core coverage 82.12%**. Ruff/format/lock PASS po usunięciu lokalnego B904. Fingerprints Pyright przed/po identyczne: 123 błędy commands.py, delta 0; utils i nowy test 0 błędów. Przy pomiarze changed-code coverage dodano małe in-process kontrole helpera/handlera (procesowe dzieci nie raportowały coverage); pierwsza próba tych nowych kontroli miała 4 błędy fixture przez pusty chain input, poprawiono test na poprawny syntetyczny input bez zmiany produktu. Następny krok: powtórzyć końcowe packs/coverage i zamknąć niezależny review. Evidence: `green.log`, `type-delta.json`, `full.log` w tym samym scratch.
4. **QW1 końcowe bramki:** po poprawce fixture final focused **201 PASS**, full **1345 PASS / 1 skipped**, core coverage **82.12%**; nowy pack ma 32 próby procesowe + 9 kontroli in-process. Changed executable statements: utils **13/13**, commands **22/22**, razem **100%** (`changed-coverage.json`). Ruff repo check, format check, lock offline i diff check PASS; utils/test strict Pyright 0 błędów. Skan dodanych linii produkcyjnych bez trafień secret/shell/eval/pickle/SQL. Zmieniono wyłącznie handler, helper, nowy test, developer guide/changelog i tracker; bez staging/commit/push/providerów live. Końcowy review niezależny w toku — status nie jest jeszcze completed.
5. **QW1 review i closeout:** niezależny review `deleg_ffa45ff8` PASS: brak blocking logic/security findings. Zweryfikowano sugestie względem aktualnego testu; in-process UnicodeError i parent-unwritable już były, ale wzmocniono procesową macierz o UnicodeEncodeError, missing-target/unwritable-parent i stat PermissionError (oba entrypointy × JSON/text). Error assertions sprawdzają cały envelope/dokładny bounded komunikat oraz brak output content. Bez dalszych zmian produkcyjnych. Końcowy focused **213 PASS**, full **1357 PASS / 1 skipped**, core **82.12%**; 44 próby procesowe + 9 in-process. Ruff/format, strict utils/test Pyright, offline lock i diff PASS. Rozdzielono pliki coverage runnerów, aby równoległe packs nie nadpisywały tego samego store. QW1 ukończony lokalnie; nie kwalifikuje całej rodziny chain jako agent-ready.

## QW1 inventory lokalnego closeoutu

Dokładny numstat przed rozpoczęciem QW2 (plan zawiera również intake pozostałych QW):
- `cli/commands.py`: +36 / -10
- `cli/utils.py`: +21 / -0
- `docs/CHANGELOG.md`: +1 / -0
- `docs/README-DEV.md`: +15 / -0
- nowy `tests/test_prompt_chain_output_preflight.py`: +365 / -0
- nowy tracker: liczba linii zależy od dalszych wpisów; brak stagingu, nie jest w zwykłym `git diff --numstat`.

Stan przed delivery: wszystkie zmiany w WSL checkout na baseline; Windows checkout nie modyfikowano. Nie wykonano zmian CI ani modeli live. Logs i coverage: `/home/voytas/.hermes/cache/scratch/promptmanager-qw1-tp2wlqu2/` (scratch ma ograniczoną retencję; trwały checkpoint to ten tracker).

6. **QW1 delivery approval:** użytkownik zatwierdził commit/push QW1, a następnie QW2. Remote master zgodny z baseline, standard protection brak (404), rules/branches/master puste. Stage tylko sześć plików QW1; bez zmian polityk. Nowa zgoda przed QW3 i osobna zgoda na publikację QW2.
7. **QW1 delivery / QW2 intake:** commit `cd0a3a591d29c1737e7aac5583ebc7713c00c210`, +550/-10, sześć plików; push do master bez protection bypass. Po fetch HEAD=origin/master=ls-remote, clean tree; remote tracker readback potwierdzony. [Quality Gates 36757823792](https://github.com/voytas75/PromptManager/actions/runs/36757823792) PASS (Ruff, Pyright main/config/models, pytest coverage i clean tree); CodeQL brak w exact-SHA run/check list, nie nazywać go PASS. Banner o 8 vulnerabilities odłożony bez triage. QW2 baseline ten commit, kontrakt receipt/error określony przed RED; providers/live, CI/settings, QW3 semantics i commit/push QW2 poza zakresem.
8. **QW2 RED/GREEN:** rozszerzono istniejący pack procesowy o typed/unexpected runner exceptions, zachowanie istniejącego/brak nowego artefaktu, returned success/partial/failed JSON file i no-file oraz text legacy controls. RED **22 failed / 59 PASS**, expected text receipt/error/nonzero mismatch bez błędu fixture; po minimalnej zmianie handlera **81 PASS**. Zmieniono receipt i tylko JSON runner-error translation; full artifact/no-file JSON/text/returned failed exit nie zmieniono. Logs: `/home/voytas/.hermes/cache/scratch/promptmanager-qw2-3xeduz64/red.log`, `green.log`.
9. **QW2 bramki:** dopisano in-process exception controls (coverage) i wzmocniono sąsiedni file receipt test. Focused **247 PASS**, full provider-free **1391 PASS / 1 skipped**, core **82.12%**, added executable statements **9/9**. Ruff/format/lock/diff PASS; testy strict Pyright 0, commands inherited123 fingerprints bez nowych. JSON przykład docs parse PASS, skan produkcyjnych dodanych linii bez trafień. Jedno sprawdzenie shellowe miało błąd quoting przez markdown backtick, powtórzono programatycznie bez zmiany produktu. Scope: commands, dwa testy, developer guide/changelog/tracker; bez QW3/provider/CI/Git delivery. Niezależny review `deleg_e0c6f3b4` w toku.
10. **QW2 review / closeout:** `deleg_e0c6f3b4` PASS bez blocking logic/security findings. Sugestie zweryfikowano z aktualnym plikiem: uwaga o discovery/RED w trackerze odnosiła się do starszego odczytu; przed zamknięciem tracker już zawierał osobne wyniki QW1/QW2 i GREEN. Dodano trwałe procesowe regresje dla trzech runner-error types bez output-file oraz względnych/absolutnych Unicode paths ze spacją, z readback rzeczywistego artefaktu. Bez nowych zmian produkcyjnych. Końcowy focused **257 PASS**, full provider-free **1401 PASS / 1 skipped**, core **82.12%**; 82 procesowe + 15 in-process w głównym packu. Changed statements 9/9 pokryte. Ruff/format/lock/diff i testy strict Pyright PASS. QW2 ukończony lokalnie; HEAD nadal `cd0a3a5`, sześć plików QW2 dirty, nie staged/committed/pushed. Scope wyjścia/runnera nie kwalifikuje całego CLI jako agent-ready.

## QW2 inventory lokalnego closeoutu

Baseline: `cd0a3a5`. Dokładny numstat poza samym trackerem:
- `cli/commands.py`: +19 / -1
- `docs/CHANGELOG.md`: +1 / -0
- `docs/README-DEV.md`: +19 / -3
- `tests/test_prompt_chain_cli.py`: +8 / -1
- `tests/test_prompt_chain_output_preflight.py`: +144 / -2

Tracker obejmuje również potwierdzenie dostarczenia QW1 i jego własny numstat zmienia się przy dalszych wpisach. Logs: `/home/voytas/.hermes/cache/scratch/promptmanager-qw2-3xeduz64/` (retencja scratch ograniczona). W closeout QW2 nie wykonano modeli live, zmian CI/deps, Windows checkout ani QW3. Publikacja pozostaje niezatwierdzona; późniejsza zgoda na QW3 jest zapisana poniżej.

11. **QW3 zgoda/intake:** użytkownik zatwierdził QW3 po zamknięciu QW2. HEAD `cd0a3a5`; sześć lokalnych plików QW2 (+214/-14) zachować bez stagingu/commitu/pusha. Scope: dwa istniejące handlery chain/benchmark, ich testy, developer guide/changelog i ten tracker; bez flags/benchmark JSON/retries/engine/CI/deps/providers/Windows/QW4. Odczyt backendu wykazał także `skipped` i no-step -> `failed`; discovery obejmuje te gałęzie. Niezależny read-only compatibility check `deleg_432687f6` aktywny; mapowanie zostanie zapisane przed kodem.
12. **QW3 discovery/kontrakt:** `deleg_432687f6` nie znalazł blokującego klienta; parent potwierdził pass-through entrypoint (`cli/entrypoint.py:54–56`) i backend `chains.py:384–404`, benchmark sentinel `execution_history.py:99,103–107`. Kontrakt 0 tylko dla pełnego sukcesu, pozostałe 5 zapisany powyżej przed kodem. Zachować stdout i artefakty partial/failed; brak nowego error envelope dla zwróconego wyniku. Baseline Pyright dotkniętych plików **123** błędy, wszystkie odziedziczone commands.py; snapshot QW2 i logs poza repo: `/home/voytas/.hermes/cache/scratch/promptmanager-qw3-ov383ihc/`. Następny krok RED przez oba entrypointy i in-process coverage controls, potem minimalny handler-only GREEN.
13. **QW3 RED/GREEN:** pierwszy sekwencyjny pack osiągnął limit narzędzia 420 s, nie był pełnym wynikiem; po potwierdzeniu braku pozostałych procesów powtórzono z czterema workerami. RED **98 failed / 151 PASS**, błędy exit 0 zamiast 5. Dodatkowy in-process pack po usunięciu błędnej asercji o nagłówku `Status:` w default text dał **20 failed / 4 PASS** (fixture nie uzasadnia zmiany rendererów). Minimalny GREEN: **273 PASS**. Zmiana produkcyjna tylko dwa handlery; domain exit wyznaczony przed renderem i zwracany po zapisie/stdout. Dodano oba entrypointy, JSON/full receipt, wszystkie text selectors × file/no-file × success/partial/failed, skipped/empty/unknown JSON, mixed/all-errors/no-runs/empty-error benchmark i in-process coverage controls. Ruff repo/format i trzy testy strict Pyright PASS; offline lock/diff PASS. Docs/changelog zsynchronizowane. Następny krok pełne bramki, dokładna delta typów/coverage/scope i niezależny review.
14. **QW3 końcowe bramki:** focused **405 PASS**, pełny provider-free suite **1548 PASS / 1 skipped**, core **82.12%**. Changed executable statements **10/10**; output exit branches pokryte in-process, procesy potwierdzają stdout/stderr/calls/artefakty. Ruff repo check/format, configured Pyright 0 i strict trzy testy 0; commands.py nadal **123** odziedziczone błędy, dokładna delta fingerprintów **0**. Lock offline/diff PASS; security scan dodanych produkcyjnych linii bez trafień. Brak stagingu/commitu/pusha/modeli live/Windows/QW4. `deleg_c656d822` review aktywny; trwały closeout nastąpi po werdykcie. Logs/snapshot/type delta/coverage w `/home/voytas/.hermes/cache/scratch/promptmanager-qw3-ov383ihc/` (retencja scratch ograniczona).
15. **QW3 review/closeout:** `deleg_c656d822` PASS bez blocking logic/security findings. Parent sprawdził sugestie względem aktualnych plików i dodał 16 regresji process `--status-only`/`--compact` × empty/unknown × file/no-file × oba entrypointy oraz kontrolę benchmark `error=None` z pustym preview (exit 0, `(empty response)`). Bez kolejnych zmian produkcyjnych. Końcowy focused **422 PASS**, full provider-free **1565 PASS / 1 skipped**, core **82.12%**, changed statements **10/10**. Ruff repo/format, strict trzy testy, offline lock/diff PASS. Typy produkcyjne bez zmian od poprzednich bramek: inherited123, delta0; configured Pyright PASS. QW3 ukończony lokalnie, HEAD nadal `cd0a3a5`; siedem plików combined QW2/QW3 dirty, bez stagingu/commitu/pusha/providers/Windows/QW4. Klienci spoza checkoutu pozostają do weryfikacji; pełnej agent-ready kwalifikacji nie wykonano.

## QW3 inventory względem zamkniętego QW2

Dokładny numstat przed końcowym dopisaniem wyników do samego trackera:
- `cli/commands.py`: +10 / -9
- `docs/CHANGELOG.md`: +2 / -1
- `docs/README-DEV.md`: +26 / -2
- `tests/test_cli_benchmark_output.py`: +54 / -6
- `tests/test_prompt_chain_cli.py`: +47 / -4
- `tests/test_prompt_chain_output_preflight.py`: +158 / -9

Tracker ma również zgodę/discovery i aktualizacje checkpointów, jego numstat zmienia się przy każdym wpisie. Względem HEAD repo zawiera zarówno QW2, jak i QW3; nie określać combined diff jako samego QW3. Publikacja obu pozostaje niezatwierdzona.

16. **QW4 zgoda/discovery:** użytkownik zatwierdził QW4; HEAD `cd0a3a5`, siedem wcześniejszych dirty plików combined QW2/QW3 +527/-36 zachować. Odczyt `main.py:309–314` i factory `:424–431` potwierdził istniejący opt-out na expected offline warning/notify bez zmiany manager state. Example logging StreamHandler kieruje WARN na stdout; fallback INFO na stderr. Scope przewidywany: `main.py` (JSON-only bootstrap, kilka linii), nowy wąski pack procesowy/in-process, developer guide/changelog/tracker; nie zmieniać loggerów/factory/CommandSpec defaults/parser/CI/deps/providers/Windows ani QW5. Snapshot i type baseline w `/home/voytas/.hermes/cache/scratch/promptmanager-qw4-g0bq6h7s/`; main0/commands inherited123. Następny krok tests RED przed produkcją.
17. **QW4 RED/GREEN:** initial pack wykrył brak obowiązkowego description w fixture Prompt; to błąd harnessu, nie produktu. Po poprawce fixture RED **22 failed / 20 PASS** — oczekiwane warning/notify w JSON, także stdout example logging; text i startup-error controls już PASS. Minimalny bootstrap +2 linie oraz update docstring historii; GREEN **42 PASS**: 36 process + 6 in-process controls, oba entrypointy × fallback/example logging × JSON/text, state/reason/notify, unrelated warning, startup failure i render missing variable. W procesach prawdziwa factory i metoda set_llm_status, syntetyczny manager/repository bez SQLite/Chroma, socket fail guard; nie kwalifikuje real-store bootstrapu jako bez efektów ubocznych. Ruff repo/format, strict main/test, offline lock/diff PASS. Docs/changelog zsynchronizowane. Następny krok pełne provider-free bramki, exact type/coverage/scope i read-only review.
18. **QW4 bramki/review checkpoint:** dodano cztery process controls `--validate-only` JSON (bez zmiany produkcji). Focused **190 PASS**, changed executable statements **2/2**. Repo Ruff/format, configured Pyright i strict main/test PASS; main0, commands inherited123 przed/po, fingerprint delta0. Security scan added production clean, offline lock/diff PASS. Pierwszy full suite osiągnął limit 400 s bez końcowego podsumowania; nie raportować PASS, potwierdzono brak pozostawionego pytest. Powtórzony pełny provider-free suite działa jako `proc_0ba04950bdfe`, notify-on-complete, wyniki w scratch. Review `deleg_5dffd517` aktywny. Baseline przed QW4 w scratch pozwala oddzielić wcześniejsze QW2/QW3; nowe pliki nie stage'owane. Bez providers/CI/deps/Git delivery/Windows/QW5.

## Końcowe checkpointy i inventory

24. **QW2–QW5 delivery intake:** użytkownik zatwierdził commit/push ukończonych QW2–QW5. Scope: dziesięć dirty paths (osiem tracked i dwa nowe testy), bez innych zmian. Fetch potwierdził HEAD=origin/master=`cd0a3a5`, brak outgoing/incoming commits; protection endpoint zwrócił Branch not protected, branch rules `[]`. Real suite evidence 1667 PASS/1 skipped i core82.12%, review każdego QW PASS; produkcja i wcześniejsze code/tests zachowane względem zakończonych checkpointów. Zatwierdzony zwykły push na master, bez force/policy/deps/security/Windows zmian. Remote/CI jeszcze niepotwierdzone; aktualizację delivery docs wykonać po weryfikacji produktu.

20. **QW5 zgoda/discovery:** użytkownik „proceduj QW5”; HEAD nadal `cd0a3a5`, dziewięć wcześniejszych dirty plików zachowano w scratch baseline `/home/voytas/.hermes/cache/scratch/promptmanager-qw5-ypqvm94d/`. Odczyt potwierdził installed dispatch `cli/entrypoint.py:30–33`, module brak przed heavy imports, za to późny `_run_application` już ma handler. Settings/parser i bezpośredni SQLite handler mogą działać bez core/GUI/provider importów. Zakres: `main.py` około +5 (historia+dispatch), jeden nowy pack procesowy około 200–300 linii, README-dev/changelog/tracker; bez parser/handler/SQL/CI/deps/config/provider/Windows/Git delivery zmian. Baseline main0/commands123. Testy mają użyć obu rzeczywistych entrypointów z import/network fail guard i syntetyczną bazą, preview/error/apply + backup/readback i zachowane inne ścieżki. Alternatywa — używać tylko installed launcher — zostawia modułowy kontrakt niespójny; zmienić zakres dopiero gdy test pokaże konieczną zależność handlera od heavy runtime. Następny krok RED.

Końcowy inventory bez trackera: `main.py` +3/-0 (historia +1, produkcja +2), `docs/README-DEV.md` +16/-0, `docs/CHANGELOG.md` +1/-0, nowy `tests/test_cli_offline_json_announcements.py` +321/-0. Tracker obejmuje intake i dalsze checkpoints, jego numstat zależy od dalszych wpisów. Względem HEAD diff zawiera QW2/QW3/QW4; publikacja nadal niezatwierdzona.

19. **QW4 closeout:** review `deleg_5dffd517` PASS, security/logic findings puste. Sugestie przyjęte: unrelated-warning test używa teraz rzeczywistego `factory.factory_logger`, bootstrap-only settings są `model_construct` bez odczytu env/dotenv/JSON; produkcja bez dalszych zmian. Historyczne 42 PASS zachowane w wpisie 17; pack po validate-only ma 46 przypadków. Review podał własne 46 PASS + 12 dodatkowych probes, nie wliczać ich do parent gates. Powtórka background `proc_0ba04950bdfe` rzeczywiście nie wystartowała (exit127 względnego interpretera), więc wcześniejszy checkpoint o działaniu nie jest wynikiem suite. Poprawny rerun z absolute interpreter zakończył się exit0: **1611 PASS / 1 skipped**, core **82.12%**, 372.94 s; focused po sugestiach **190 PASS**. Ruff repo/format, configured i strict main/test Pyright, lock offline i diff PASS; executable delta **2/2 covered**. Zapisane logi/coverage/type-delta w scratch. HEAD nadal QW1; bez staging/commit/push, providers live, CI/deps zmian i QW5. QW4 zamknięty lokalnie; następny krok wyłącznie pytanie o QW5.

21. **QW5 RED/GREEN:** pack rzeczywistych entrypointów RED **16 failed / 22 PASS**; wszystkie runtime module edit cases przerwane na zabronionym `cli.commands` przed handlerem, installed/help/unrelated controls już PASS. Minimalna produkcja `main.py` +5/-0: historia+1 i czteroliniowy dispatch jak pozostałe early routes; GREEN **38 PASS**. Preview i błędne dane bez zmian bajtów bazy; apply/backup/readback oraz body/description/version/activity zachowane; brak heavy imports/network i Chroma. Dodano dwa in-process controls exact args/exit dla changed-code coverage. Docs/changelog zsynchronizowane, full/focused/typy i niezależny review jeszcze w toku. Zakres pozostaje bez parser/handler/SQL/provider/config/CI/deps/Git delivery zmian.

22. **QW5 bramki checkpoint:** final pack ma 40 testów; focused **178 PASS**, full **1651 PASS / 1 skipped**, core **82.12%**, exit0 w 357.85 s. Strict main/test oraz configured Pyright PASS; main0/commands inherited123 przed/po fingerprint delta0; repo Ruff/format, lock offline/diff PASS. Coverage added statements **3/3**, security added production clean. Scope vs QW4: main +5/-0, README-dev +15/-0, changelog +1/-0, nowy test +300/-0, tracker osobno. Porównanie baseline potwierdziło wcześniejsze code/tests QW2–QW4 byte-unchanged. Review `deleg_2fa6dc65` aktywny; zamknięcie jeszcze niepotwierdzone. Nie wykonano stage/commit/push, providerów live, CI/deps/config/Windows zmian.

23. **QW5 review/closeout:** niezależny review `deleg_2fa6dc65` PASS; security/logic puste. Potwierdził 40 tests, 16 scratch probes i 44 niezmienione inne routes — wyniki review nie dodawane do parent totals. Przyjęto sugestie: dokładny blocked `cli.commands` dla unrelated route oraz trwałe module/installed × JSON/text controls backup collision, pending WAL, no-op apply i valid references. W nowym harnessie no-op zostawiał WAL i poprawnie otrzymał CATALOG_BUSY (4 failed/52 PASS); dodano checkpoint wyłącznie fixture przed oczekiwanym no-op, bez zmiany produktu, potem **56 PASS**. Końcowe powtórzone bramki: focused **194 PASS**, full **1667 PASS / 1 skipped**, core **82.12%**, exit0 w 409.87 s; Ruff/format, configured i strict main/test Pyright, lock/diff PASS, changed statements **3/3**. Produkcja nadal +5/-0; docs +15/-0 i +1/-0; test +374/-0, tracker oddzielnie. Wcześniejsze QW2–QW4 code/tests bez zmian. QW5 zamknięty lokalnie, cały plan wykonany; QW2–QW5 bez stage/commit/push i providers live, bez CI/deps/config/Windows zmian. Next: wyłącznie decyzja o publikacji.
