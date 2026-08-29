import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

DELIMITER = "#####"


def generate_key(password):
    key = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(key)


def encrypt_message(message, password):
    cipher = Fernet(generate_key(password))
    encrypted_message = cipher.encrypt(message.encode("utf-8"))
    return encrypted_message.decode("utf-8")


def decrypt_message(message, password):
    cipher = Fernet(generate_key(password))
    decrypted_message = cipher.decrypt(message.encode("utf-8"))
    return decrypted_message.decode("utf-8")


def text_to_binary(text):
    return "".join(format(ord(char), "08b") for char in text)


def encode_text(image, message, password=""):
    image = image.convert("RGB")

    if password and password.strip():
        message = encrypt_message(message, password)

    message += DELIMITER
    binary_message = text_to_binary(message)

    width, height = image.size
    max_capacity = width * height * 3

    if len(binary_message) > max_capacity:
        raise ValueError("Message is too large for this image.")

    pixels = image.load()
    index = 0

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]

            if index < len(binary_message):
                r = (r & 254) | int(binary_message[index])
                index += 1

            if index < len(binary_message):
                g = (g & 254) | int(binary_message[index])
                index += 1

            if index < len(binary_message):
                b = (b & 254) | int(binary_message[index])
                index += 1

            pixels[x, y] = (r, g, b)

            if index >= len(binary_message):
                return image

    return image


def decode_text(image, password=""):
    image = image.convert("RGB")

    pixels = image.load()
    width, height = image.size

    bits = []

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]

            bits.append(str(r & 1))
            bits.append(str(g & 1))
            bits.append(str(b & 1))

    chars = []

    for i in range(0, len(bits), 8):
        byte = "".join(bits[i:i + 8])

        if len(byte) < 8:
            break

        chars.append(chr(int(byte, 2)))
        text = "".join(chars)

        if text.endswith(DELIMITER):
            message = text[:-len(DELIMITER)]

            if message.startswith("gAAAAA"):
                if not password or not password.strip():
                    return "🔐 This message is password protected. Please enter the password."

                try:
                    return decrypt_message(message, password)

                except InvalidToken:
                    return "❌ Wrong password. Please try again."

                except Exception:
                    return "❌ Unable to decrypt the message."

            return message

    return "❌ No hidden message found in this image."
