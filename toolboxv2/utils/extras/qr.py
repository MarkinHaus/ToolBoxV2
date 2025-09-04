import sys



def print_qrcode_to_console(content):
    try:

        qrcode = __import__('qrcode')
        qr = qrcode.main.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_Q,
            box_size=1,
            border=2,
        )
        qr.add_data(content)
        qr.make(fit=True)
        print("QR: ", content)
        if 'unittest' not in sys.argv[0]:
            qr.print_ascii(invert=True)
        return qr
    except ImportError:
        print("QRCode is not available run pip install qrcode")
        return None
