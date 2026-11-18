import asyncio
import signal
import sys

import structlog

from app.config import settings, configure_logging
from app.database import init_db, close_db
from app.telegram import TelegramBot
from app.agent.background import BackgroundTasks


configure_logging()

logger = structlog.get_logger()


class Application:
    def __init__(self):
        self.bot: TelegramBot | None = None
        self.background_tasks: BackgroundTasks | None = None
        self._shutdown_event = asyncio.Event()
        self._initialized = False
        self._running = False

    async def initialize(self):
        logger.info("initializing_application")

        await init_db()
        logger.info("database_initialized")

        self.bot = TelegramBot(
            settings.telegram_bot_token
        )

        await self.bot.initialize()

        self.background_tasks = BackgroundTasks(
            self.bot
        )

        await self.background_tasks.start()

        self._initialized = True

        logger.info(
            "application_initialized"
        )

    async def run(self):
        if not self.bot:
            raise RuntimeError(
                "Bot has not been initialized"
            )

        logger.info("starting_bot")

        await self.bot.start()

        self._running = True

        logger.info(
            "application_running"
        )

        try:
            # Keep asyncio loop alive.
            while not self._shutdown_event.is_set():
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info(
                "application_run_cancelled"
            )
            raise

    async def shutdown(self):
        logger.info("shutting_down")

        # Stop background tasks first.
        if self.background_tasks:
            try:
                await self.background_tasks.stop()
            except Exception as e:
                logger.error(
                    "background_tasks_shutdown_failed",
                    error=str(e),
                )

            self.background_tasks = None

        # Stop Telegram bot.
        if self.bot:
            try:
                await self.bot.stop()
            except Exception as e:
                logger.error(
                    "bot_shutdown_failed",
                    error=str(e),
                )

            self.bot = None

        # Close database last.
        try:
            await close_db()
        except Exception as e:
            logger.error(
                "database_shutdown_failed",
                error=str(e),
            )

        self._running = False
        self._initialized = False

        logger.info(
            "shutdown_complete"
        )

    def signal_handler(self, sig):
        logger.info(
            "signal_received",
            signal=sig,
        )

        self._shutdown_event.set()


async def main():
    app = Application()

    # Windows does not support add_signal_handler().
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()

        for sig in (
            signal.SIGTERM,
            signal.SIGINT,
        ):
            loop.add_signal_handler(
                sig,
                app.signal_handler,
                sig,
            )

    try:
        await app.initialize()

        await app.run()

    except asyncio.CancelledError:
        logger.info(
            "main_cancelled"
        )

    except KeyboardInterrupt:
        logger.info(
            "keyboard_interrupt"
        )

    except Exception as e:
        logger.error(
            "application_error",
            error=str(e),
            exc_info=True,
        )

        # Do not re-raise here. The finally block will perform
        # the cleanup and then the process exits cleanly.
        #
        # Re-raising was not useful in this situation because
        # it caused the shutdown exception to obscure the
        # original Telegram connection problem.

    finally:
        await app.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass