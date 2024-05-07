import qrcode


def print_qrcode_to_console(content):
    qr = qrcode.main.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=1,
        border=2,
    )
    qr.add_data(content)
    qr.make(fit=True)
    print("QR: ", content)
    qr.print_ascii(invert=True)
    return qr
