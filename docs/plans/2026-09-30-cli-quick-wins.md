# PromptManager — CLI quick wins

Status: aktywny — QW1 commit/push zatwierdzony; QW2 zatwierdzony po dostarczeniu QW1. QW3–QW5 wymagają osobnych zgód.
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

Status: ukończony lokalnie — implementacja, bramki, docs i niezależny review PASS; commit/push zatwierdzony, dostarczenie w toku.
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

Status: zatwierdzony — rozpocząć po zweryfikowanym dostarczeniu QW1.
Zakres: handler chain i renderer receipt, odpowiednie testy procesowe, developer guide/changelog/tracker.
- `--json --output-file`: jeden JSON receipt z identyfikatorem komendy, ścieżką artefaktu i statusem run zamiast tekstowego `Saved ...`; dokładny schema ustalić przed RED.
- Wyjątki chain w JSON: sanitarny error JSON stderr, pusty stdout, zachowane kody błędów. Nie echo surowego provider exception.
- Zachować default text, payload JSON bez pliku i run-status semantics do QW3.
Done: sukces/wyjątek/file-mode z obu entrypointów, readback artefaktu, marker privacy i bramki. Parser/startup całej rodziny nie jest domyślnie zakresem QW2.

## QW3 — exit code a domain success

Status: niezatwierdzony — wymagana nowa zgoda i jawna decyzja kompatybilności.
Zakres: istniejące chain/benchmark handlers, testy sukces/partial/failure, docs/changelog/tracker.
- Przed kodem ustalić mapowanie success/partial/failed/no-runs do exit oraz zidentyfikować istniejące oczekiwania klientów/testów.
- Failed chain i benchmark all-errors nie powinny udawać sukcesu run. Wariant węższy: opt-in strict status, jeżeli zmiana domyślnego exit łamie potwierdzonego klienta.
- Zachować materiał diagnostyczny i zapis artefaktu nieudanego run tam, gdzie jest częścią istniejącego kontraktu. Nie dodawać benchmark JSON ani nowych retries w tym QW bez zgody.
Done: kontrakt zaakceptowany, deterministyczne run outcome fixtures i real-process exit/output parity, docs i bramki.

## QW4 — oczekiwane offline warnings w wybranych JSON success

Status: niezatwierdzony — wymagana nowa zgoda.
Zakres: CommandSpec/bootstrap flags tylko dla `prompt-render --json` i `tag-list --json`, odpowiednie testy, docs/changelog/tracker.
- Użyć istniejącej kontroli `announce_offline_llm`; nie wyciszać błędów/globalnych loggerów ani zmieniać zwykłego text-mode.
- Próby ze scratch CWD i repozytoryjną przykładową konfiguracją logowania.
Done: jeden JSON stdout, pusty stderr na oczekiwanym offline sukcesie w obu entrypointach; default text bez regresji. Nie kwalifikuje bezpieczeństwa renderer input ani parser/startup/runtime errors całej rodziny.

## QW5 — early dispatch prompt-edit parity

Status: niezatwierdzony — wymagana nowa zgoda.
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

- Bramka następnego zadania: zweryfikować commit/push QW1; zgoda QW2 uzyskana, jeszcze bez zmian QW2.
- Potwierdzone: końcowy focused 213 PASS, full 1357 PASS / 1 skipped, core coverage 82.12%; changed executable statements 35/35 = 100%; Ruff/format/lock/diff PASS. Pyright utils/test 0, commands.py odziedziczone 123 bez nowych fingerprintów. Niezależny review PASS, wszystkie trzy sugestie testowe uwzględnione.
- Następny krok: commit/push QW1, exact-SHA CI, następnie QW2 (JSON receipt i sanitized chain runtime errors). Publikacja QW2 wymaga osobnej zgody. Preflight jest best-effort, nie rezerwuje późniejszego zapisu i nie chroni przed bootstrapem managera.

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
