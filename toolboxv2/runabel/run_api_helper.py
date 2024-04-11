NAME = "api"


def stest_from_js(payload=None):
    print(f"Called from js with payload: {payload}")
    return "done"


def run(_, _0):
    from toolboxv2.mods.EventManager.module import EventManagerClass, SourceTypes, Scope
    from toolboxv2 import tbef

    if _0.host or _0.port:
        _.run_any(tbef.API_MANAGER.EDITAPI, api_name=_0.name, host=_0.host if _0.host else "localhost",
                  port=_0.port if _0.port else 5000)
    r = _.run_any(tbef.API_MANAGER.STARTAPI, api_name=_0.name, live=not _.debug, reload=_.debug)
    print(r)
    _.run_any(tbef.EVENTMANAGER.START_WEB_EVENTS)

    ev: EventManagerClass = _.run_any(tbef.EVENTMANAGER.GETEVENTMANAGERC)

    ev.identification = "P0"
    event = ev.make_event_from_fuction(stest_from_js, "event_fuction", source_types=SourceTypes.P)
    ev_id = event.event_id
    event.scope = Scope.local
    print("event ID :", str(ev_id))
    ev.register_event(event)

    try:
        _.run_runnable('TBtray')
    except:
        _.run_runnable('cli')


if __name__ == "__main__":
    import qrcode

    qr = qrcode.main.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=1,
        border=2,
    )
    qr.add_data('Some data')
    qr.make(fit=True)

    qr.print_ascii(invert=True)
    # img.
