"""F3: notificaciones desacopladas (n8n/Telegram fuera del hilo de deteccion).

Cubre los 8 criterios pedidos:
1. Enqueue no bloqueante (el pipeline de camara no espera a la red).
2. El worker ejecuta la notificacion.
3. Error de webhook: el worker/camara siguen vivos y el siguiente trabajo corre.
4. Error de Telegram: igual.
5. Cola llena: politica definida, productor no se bloquea.
6. Multiples camaras (productores concurrentes).
7. Shutdown limpio.
8. Sanitizacion de tokens en logs.

Solo usa la biblioteca estandar (sin cv2/torch/ultralytics/requests): la red se
inyecta con callbacks. Corren SIEMPRE en local."""

import queue
import threading
import time

import pytest

from notifications import HourClock, NotificationWorker, sanitize_token
from testutil import wait_until


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _recorder(*args, **kwargs):
    """Devuelve un callable que registra las llamadas. `sleep_secs` simula una
    request lenta (bloqueo de red)."""
    calls = []

    def fn(*a, **kw):
        calls.append((a, kw))
        time.sleep(getattr(fn, "sleep_secs", 0.0))
        return True

    fn.calls = calls
    return fn


@pytest.fixture()
def worker_factory():
    """Crea workers de notificaciones aislados por test (con callbacks fake).
    Inyecta SIEMPRE ambos callbacks de red y un webhook_url, para que el test
    sea determinista y no toque requests ni env."""
    created = []

    def make(**overrides):
        kw = dict(
            webhook_url="http://n8n.test/webhook",
            telegram_token="123:fake",
            telegram_chat_id="123",
        )
        kw.update(overrides)
        # si no se dieron explicitos, inyecto recorders
        kw.setdefault("send_webhook_fn", _recorder())
        kw.setdefault("send_telegram_fn", _recorder())
        w = NotificationWorker(
            queue_size=kw.pop("queue_size", 8),
            num_workers=kw.pop("num_workers", 1),
            **kw,
        )
        w.start()
        created.append(w)
        return w

    yield make
    for w in created:
        w.stop()
        w.join(timeout=3.0)


# ---------------------------------------------------------------------------
# 1. Enqueue NO bloqueante
# ---------------------------------------------------------------------------

def test_enqueue_no_bloquea_el_productor(worker_factory):
    """Aunque el consumidor duerma varios segundos (request lenta), enqueue
    devuelve al instante y el productor no espera a la red."""
    webhook = _recorder()
    webhook.sleep_secs = 2.0
    w = worker_factory(send_webhook_fn=webhook)

    t0 = time.perf_counter()
    ok = w.submit_hourly_report("cam", "2026-08-31 10:00", 3, 2, jpeg=b"x")
    elapsed = time.perf_counter() - t0

    assert ok is True
    # el enqueue no debe tardar ni un segundo (el worker duerme 2s procesando)
    assert elapsed < 1.0


# ---------------------------------------------------------------------------
# 2. El worker ejecuta la notificacion
# ---------------------------------------------------------------------------

def test_worker_ejecuta_la_notificacion(worker_factory):
    webhook = _recorder()
    telegram = _recorder()
    w = worker_factory(send_webhook_fn=webhook, send_telegram_fn=telegram)

    w.submit_hourly_report("cam", "2026-08-31 11:00", 5, 4, jpeg=b"foto")

    assert wait_until(lambda: webhook.calls, timeout=3.0)
    assert wait_until(lambda: telegram.calls, timeout=3.0)
    # verifico el payload del webhook (firma: url, payload, timeout)
    url, payload, timeout = webhook.calls[0][0]
    assert payload["camera"] == "cam"
    assert payload["hour"] == "2026-08-31 11:00"
    assert payload["in_count"] == 5
    assert payload["out_count"] == 4
    # el txt de telegram contiene los mismos numeros y lleva la foto
    args, kwargs = telegram.calls[0]
    tok, chat, text = args[:3]
    assert "11:00" in text and "5" in text
    # telegram recibe la foto en photo_bytes (kwargs)
    assert kwargs.get("photo_bytes") == b"foto"


# ---------------------------------------------------------------------------
# 3. Error de webhook
# ---------------------------------------------------------------------------

def test_error_webhook_no_mata_al_worker_ni_detiene_la_cola(worker_factory):
    webhook = _recorder()
    telegram = _recorder()
    # provoca un error HTTP en la primera llamada del webhook
    def failing_webhook(url, payload, timeout):
        if not getattr(failing_webhook, "failed", False):
            failing_webhook.failed = True
            raise TimeoutError("timeout en webhook a n8n")
        return webhook(url, payload, timeout)

    w = worker_factory(
        send_webhook_fn=failing_webhook,
        send_telegram_fn=telegram,
    )

    # el primer trabajo (cam-a) falla en el webhook
    w.submit_hourly_report("cam-a", "2026-08-31 12:00", 1, 1, jpeg=b"a")
    # el segundo trabajo (cam-b, camara distinta -> no coalesce) se procesa despues
    w.submit_hourly_report("cam-b", "2026-08-31 12:00", 2, 2, jpeg=b"b")

    # la request de cam-b SI se proceso (el fallo de cam-a no rompio la cola)
    assert wait_until(lambda: webhook.calls, timeout=4.0)
    assert len(webhook.calls) == 1  # solo cam-b llego (cam-a fallo en su request)
    payload = webhook.calls[0][0][1]
    assert payload["camera"] == "cam-b"
    assert w.error_count >= 1  # el fallo quedo registrado pero no mato nada
    assert w.is_alive is True


# ---------------------------------------------------------------------------
# 4. Error de Telegram
# ---------------------------------------------------------------------------

def test_error_telegram_no_mata_al_worker_ni_detiene_la_cola(worker_factory):
    webhook = _recorder()
    telegram = _recorder()

    def failing_telegram(*a, **kw):
        raise ConnectionError("red caida hacia api.telegram.org")

    w = worker_factory(
        send_webhook_fn=webhook,
        send_telegram_fn=failing_telegram,
    )

    w.submit_hourly_report("cam-a", "2026-08-31 13:00", 2, 3, jpeg=b"c")
    w.submit_message("cam-b", "alerta")

    # el webhook igual se ejecuto (el fallo de telegram en cam-a no lo afecta)
    assert wait_until(lambda: webhook.calls, timeout=3.0)
    assert wait_until(lambda: w.error_count >= 2, timeout=4.0)  # 1 hourly + 1 text
    assert w.is_alive is True


# ---------------------------------------------------------------------------
# 5. Cola llena: productor no se bloquea + politica de descarte/coalescencia
# ---------------------------------------------------------------------------

def test_cola_llena_no_bloquea_al_productor_y_descarta_con_log(worker_factory):
    # consumidor muy lento para llenar la cola con facilidad
    slow = _recorder()
    slow.sleep_secs = 0.5
    w = worker_factory(queue_size=3, num_workers=1, send_telegram_fn=slow)

    start = time.perf_counter()
    accepted = 0
    # inunda la cola con muchas camaras distintas (sin coalescer) para forzar Full
    for i in range(50):
        ok = w.submit_message(f"cam-{i % 10}", f"job {i}")
        if ok:
            accepted += 1
    elapsed = time.perf_counter() - start

    # el productor nunca se bloqueo de forma significativa
    assert elapsed < 3.0
    # con 10 camaras distintas y cola 3, hubo descartes (no entraron los 50)
    assert accepted < 50
    assert w.dropped_count > 0


def test_coalescencia_misma_camara_no_duplica(worker_factory):
    """Enviar varios reportes para la misma camara antes de que la cola los
    consuma no debe duplicar: se mantiene a lo sumo una notificacion pendiente
    por camara (politica anti-duplicados / acota la cola)."""
    # worker SIN arrancar: nada se drena, inspecciono el estado de la cola
    w = NotificationWorker(
        queue_size=10,
        num_workers=1,
        webhook_url="http://n8n.test",
        telegram_token="tk",
        telegram_chat_id="ch",
        send_webhook_fn=_recorder(),
        send_telegram_fn=_recorder(),
    )

    w.submit_hourly_report("cam", "2026-08-31 14:00", 1, 1, jpeg=b"x")
    w.submit_hourly_report("cam", "2026-08-31 14:00", 9, 9, jpeg=b"y")

    # coalescencia: en la cola real solo hay UNA entrada y el pendiente por
    # camara apunta al mas reciente (in=9)
    assert len(w._queue.queue) == 1
    pend = w._pending_by_camera["cam"]
    assert pend["in"] == 9

    # el drenado efectivo manda el ultimo reporte
    w.start()
    try:
        assert wait_until(lambda: w.processed_count >= 1, timeout=3.0)
        assert w.processed_count == 1  # un solo trabajo de la camara
    finally:
        w.stop()
        w.join(timeout=3.0)


# ---------------------------------------------------------------------------
# 6. Multiples camaras (productores concurrentes)
# ---------------------------------------------------------------------------

def test_multiples_camaras_concurrentes_no_se_corrompen(worker_factory):
    w = worker_factory(queue_size=200, num_workers=2)

    errors = []

    def producer(base):
        try:
            for i in range(20):
                # cada trabajo usa una camara distinta para forzar que TODOS
                # entren a la cola (sin coalescencia) -> prueba conc. real
                ok = w.submit_hourly_report(f"cam-{base}-{i}", f"2026-08-31 {i:02d}:00", i, i)
                assert ok is True
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=producer, args=(c,)) for c in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors
    # 4 productores x 20 camaras distintas = 80 trabajos, todos en la cola
    assert wait_until(lambda: w.processed_count >= 80, timeout=5.0)
    assert len(w._queue.queue) == 0  # drenada


# ---------------------------------------------------------------------------
# 7. Shutdown limpio
# ---------------------------------------------------------------------------

def test_shutdown_limpio_no_deja_threads(worker_factory):
    w = worker_factory()
    w.submit_hourly_report("cam", "2026-08-31 15:00", 1, 1, jpeg=b"x")

    w.stop()
    w.join(timeout=3.0)

    assert w.is_alive is False or not w._threads
    # dormir un poco para asegurar que el hilo terminó
    time.sleep(0.1)
    vivos = [t for t in w._threads if t.is_alive()]
    assert not vivos


# ---------------------------------------------------------------------------
# 8. Sanitizacion de tokens
# ---------------------------------------------------------------------------

def test_sanitize_token_redacta_token_completo():
    url = "https://api.telegram.org/bot123456:ABC-DEF_ghi/sendMessage"
    out = sanitize_token(url)
    assert "123456:ABC-DEF_ghi" not in out
    assert "<REDACTADO>" in out
    # se conserva el resto de la estructura util
    assert out.startswith("https://api.telegram.org/bot<REDACTADO>")
    assert out.endswith("/sendMessage")


def test_sanitize_token_no_rompe_texto_sin_token():
    assert sanitize_token("no hay token aqui") == "no hay token aqui"
    assert sanitize_token("bot") == "bot"  # sin token real, no toca nada


def test_error_telegram_con_token_en_log_se_sanitiza(worker_factory):
    telegram = _recorder()

    def failing_telegram(*a, **kw):
        # excepcion que trae la URL con el token completo
        raise RuntimeError(
            "ConnectionError: https://api.telegram.org/bot999999:AAAAaaaa/sendPhoto"
        )

    logs = []
    import notifications

    w = NotificationWorker(
        queue_size=8,
        num_workers=1,
        send_telegram_fn=failing_telegram,
        send_webhook_fn=None,
    )

    original_warning = notifications.logger.warning
    notifications.logger.warning = lambda *a, **k: logs.append(" ".join(str(x) for x in a))
    try:
        w.start()
        w.submit_hourly_report("cam", "2026-08-31 16:00", 1, 1, jpeg=b"x")
        assert wait_until(lambda: logs, timeout=4.0)
    finally:
        notifications.logger.warning = original_warning
        w.stop()
        w.join(timeout=3.0)

    joined = "\n".join(logs)
    assert "999999:AAAAaaaa" not in joined
    assert "<REDACTADO>" in joined


# ---------------------------------------------------------------------------
# HourClock: reporte horario independiente de frames (camara caida)
# ---------------------------------------------------------------------------

def test_hour_clock_registra_y_desregistra_camaras():
    """El reloj horario mantiene callbacks por camara de forma thread-safe:
    al registrar, quedan disponibles; al desregistrar, se quitan. El reloj NO
    depende de frames (esa es justamente su funcion de respaldo)."""
    clock = HourClock(tick_seconds=0.05)
    try:
        clock._last_hour = None
        clock.register("cam-a", lambda: None)
        clock.register("cam-b", lambda: None)
        with clock._lock:
            assert set(clock._callbacks.keys()) == {"cam-a", "cam-b"}
        clock.unregister("cam-a")
        with clock._lock:
            assert set(clock._callbacks.keys()) == {"cam-b"}
    finally:
        clock.unregister("cam-a")
        clock.unregister("cam-b")

    # el reloj no arranco hilos huerfanos (no llamamos a start)
    assert clock._thread is None
