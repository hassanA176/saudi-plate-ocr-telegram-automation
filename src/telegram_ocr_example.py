from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from plate_ocr_pipeline import HybridPlateOCR

load_dotenv()

# Public portfolio example: no real fleet data is embedded.
OCR = HybridPlateOCR(reference_plates=(), use_easyocr_fallback=True)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return

    photo = update.message.photo[-1]
    telegram_file = await context.bot.get_file(photo.file_id)

    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "vehicle.jpg"
        await telegram_file.download_to_drive(custom_path=image_path)

        result = await asyncio.to_thread(OCR.recognize, image_path)

    if not result.found:
        await update.message.reply_text(
            "Could not confidently extract a plate from this image. "
            "Please use the manual fallback."
        )
        return

    await update.message.reply_text(
        f"Plate candidate: {result.plate}\n"
        f"OCR path: {result.engine}\n"
        f"Confidence: {result.confidence:.2f}"
    )


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in your local .env file.")

    application = (
        ApplicationBuilder()
        .token(token)
        .build()
    )
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.run_polling()


if __name__ == "__main__":
    main()
