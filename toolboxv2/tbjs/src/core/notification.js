// tbjs/core/notification.js
import env from './env.js';
import logger from './logger.js';


// Tauri plugin functions (will be undefined if not in Tauri or import fails)
let tauriSendNotification,
    tauriIsPermissionGrantedReal, // Actual isPermissionGranted from plugin
    tauriRequestPermission,
    tauriRegisterActionTypes,
    tauriOnAction,
    tauriCreateChannel,
    tauriDeleteChannel, // Mapped to plugin.removeChannel
    tauriListChannels,  // Mapped to plugin.channels
    tauriCancelScheduled, // Mapped to plugin.cancel (for scheduled)
    tauriRemoveActive,    // Mapped to plugin.removeActive (for displayed)
    tauriGetActive,
    tauriGetPending,
    tauriCheckPermissions,
    TauriImportanceEnum,
    TauriVisibilityEnum;

if (env.isTauri()) {
    (async () => {
        try {
            const plugin = await import('../../../simple-core/node_modules/@tauri-apps/plugin-notification/dist-js/index.js');
            tauriSendNotification = plugin.sendNotification;
            tauriIsPermissionGrantedReal = plugin.isPermissionGranted;
            tauriRequestPermission = plugin.requestPermission;
            tauriRegisterActionTypes = plugin.registerActionTypes;
            tauriOnAction = plugin.onAction;
            tauriCreateChannel = plugin.createChannel;
            tauriDeleteChannel = plugin.removeChannel;
            tauriListChannels = plugin.channels;
            tauriCancelScheduled = plugin.cancel;
            tauriRemoveActive = plugin.removeActive;
            tauriGetActive = plugin.getActive;
            tauriGetPending = plugin.getPending;
            tauriCheckPermissions = plugin.checkPermissions;

            TauriImportanceEnum = plugin.Importance || {};
            TauriVisibilityEnum = plugin.Visibility || {};
            logger.info('[Notification] Tauri notification plugin functions mapped.');
        } catch (err) {
            logger.error('[Notification] Failed to load/map Tauri notification plugin:', err);
        }
    })();
}

// For Web API, keep track of active notifications to simulate getActive/cancel
const activeWebNotifications = new Map(); // Map<tag, { notification: Notification, options: object }>

// --- Constants (from Tauri docs, useful for options) ---
export const Importance = {
    None: 0, Min: 1, Low: 2, Default: 3, High: 4,
    ...(TauriImportanceEnum || {})
};

export const Visibility = {
    Secret: -1, Private: 0, Public: 1,
    ...(TauriVisibilityEnum || {})
};

// --- Permissions ---
/**
 * Checks if permission to send notifications has been granted.
 * @returns {Promise<boolean>} True if permission is granted, false otherwise.
 */
async function isPermissionGranted() {
    if (env.isTauri() && tauriCheckPermissions) { // Use checkPermissions for a more detailed state
        try {
            const perm = await tauriCheckPermissions();
            return perm.display === 'granted';
        } catch (e) {
            logger.error('[Notification] Tauri checkPermissions error in isPermissionGranted:', e);
            // Fallback to old method if exists, otherwise false
            if (tauriIsPermissionGrantedReal) return await tauriIsPermissionGrantedReal().catch(() => false);
            return false;
        }
    }
    if (env.isWeb() && typeof window !== 'undefined' && 'Notification' in window) {
        return Notification.permission === 'granted';
    }
    logger.warn('[Notification] Platform not supported or Notification API unavailable for isPermissionGranted.');
    return false;
}

/**
 * Requests permission to send notifications.
 * @returns {Promise<boolean>} True if permission was granted, false otherwise.
 */
async function requestPermission() {
    if (env.isTauri() && tauriRequestPermission) {
        try {
            // Newer plugin's requestPermission returns 'granted' or 'denied'
            const perm = await tauriRequestPermission();
            return perm === 'granted';
        } catch (e) {
            logger.error('[Notification] Tauri requestPermission error:', e);
            return false;
        }
    }
    if (env.isWeb() && typeof window !== 'undefined' && 'Notification' in window) {
        try {
            const permission = await Notification.requestPermission();
            return permission === 'granted';
        } catch (e) {
            logger.error('[Notification] Web requestPermission error:', e);
            return false;
        }
    }
    logger.warn('[Notification] Platform not supported or Notification API unavailable for requestPermission.');
    return false;
}

/**
 * Checks the current permission state for sending notifications.
 * @returns {Promise<{ display: 'granted' | 'denied' | 'prompt' }>} Permission status.
 */
async function checkPermissions() {
    if (env.isTauri() && tauriCheckPermissions) {
        try {
            return await tauriCheckPermissions();
        } catch (e) {
            logger.error('[Notification] Tauri checkPermissions error:', e);
            return { display: 'denied' }; // Fallback
        }
    }
    if (env.isWeb() && typeof window !== 'undefined' && 'Notification' in window) {
        let permStatus = Notification.permission;
        if (permStatus === 'default') permStatus = 'prompt';
        return Promise.resolve({ display: permStatus });
    }
    logger.warn('[Notification] Platform not supported for checkPermissions.');
    return Promise.resolve({ display: 'denied' });
}


// --- Sending Notifications ---
/**
 * @typedef {Object} NotificationAction Web/Tauri
 * @property {string} id - Unique identifier for the action.
 * @property {string} title - Display text for the action button.
 * @property {string} [icon] - (Web) URL for an icon for the action button.
 * @property {boolean} [input] - (Tauri Mobile) Enables text input.
 * @property {string} [inputButtonTitle] - (Tauri Mobile) Text for input submit button.
 * @property {string} [inputPlaceholder] - (Tauri Mobile) Placeholder text for input field.
 * @property {boolean} [foreground] - (Tauri Mobile) Brings app to foreground.
 * @property {boolean} [destructive] - (Tauri Mobile) Shows action in red on iOS.
 * @property {boolean} [requiresAuthentication] - (Tauri Mobile) Requires device auth.
 */

/**
 * @typedef {Object} NotificationAttachment Tauri
 * @property {string} id - Unique identifier for the attachment.
 * @property {string} url - Content URL (e.g., 'asset:///image.png', 'file:///path/to/image.png').
 */

/**
 * @typedef {Object} NotificationScheduleOptions Tauri
 * @property {Date} [at] - Date object when the notification should be shown.
 * @property {string} [cron] - A cron string. https://docs.rs/cron/#example
 * @property {boolean} [repeats] - If the notification should be repeated.
 * @property {'millisecond' | 'second' | 'minute' | 'hour' | 'day' | 'week' | 'month' | 'year'} [every] - Interval for repeating.
 * @property {number} [count] - Number of times to repeat.
 */

/**
 * @typedef {Object} NotificationOptions
 * @property {string} title - The title of the notification.
 * @property {string} [body] - The body text of the notification.
 * @property {string} [icon] - URL to an icon image. (Tauri/Web)
 * @property {string} [sound] - (Tauri) Sound file name (without extension, placed in app's resource dir). Web: use `silent`.
 * @property {string} [tag] - (Web/Tauri) An ID for the notification. New notifications with same tag replace old ones.
 * @property {any} [data] - (Web) Custom data. For Tauri, use `extra`.
 * @property {any} [extra] - (Tauri) Custom data.
 * @property {boolean} [silent] - (Web) True to not play a sound. Default false. If `sound` is set for Tauri, this is ignored.
 * @property {boolean} [renotify] - (Web) True to make a sound/vibrate even if tag matches. Default false.
 * @property {boolean} [requireInteraction] - (Web) Keeps notification visible until user interacts. Default false.
 * @property {Array<NotificationAction>} [actions] - (Web/Tauri Mobile) Action buttons.
 * @property {Array<NotificationAttachment>} [attachments] - (Tauri) Media attachments.
 * @property {string} [channelId] - (Tauri Android) ID of the channel to use.
 * @property {NotificationScheduleOptions} [schedule] - (Tauri) Schedule options.
 * @property {number} [id] - (Tauri) A unique numeric id for the notification. Auto-generated if not provided.
 * @property {string} [summary] - (Tauri) A string to be displayed in a stack of notifications.
 * @property {string} [actionTypeId] - (Tauri Mobile) An optional action type to use for this notification.
 */

/**
 * Sends a notification.
 * @param {NotificationOptions | string} optionsOrTitle - Notification options object or title string.
 * @param {string} [bodyIfTitleString] - Body, if first arg was title string.
 * @returns {Promise<Notification | any | null>} The Notification object (Web), Tauri schedule ID, or null on failure/unsupported.
 */
async function sendNotification(optionsOrTitle, bodyIfTitleString) {
    let options;
    if (typeof optionsOrTitle === 'string') {
        options = { title: optionsOrTitle, body: bodyIfTitleString };
    } else {
        options = { ...optionsOrTitle };
    }

    const granted = await isPermissionGranted();
    if (!granted) {
        const requested = await requestPermission(); // Attempt to request if not granted
        if (!requested) {
            logger.warn('[Notification] Permission denied to send notifications.');
            return null;
        }
    }

    if (env.isTauri() && tauriSendNotification) {
        try {
            // Pass options directly; Tauri plugin handles mapping.
            // `extra` for custom data on Tauri.
            const tauriOptions = { ...options };
            if (options.data && !options.extra) {
                tauriOptions.extra = options.data;
            }
            // sendNotification in Tauri can return a schedule ID or void
            return await tauriSendNotification(tauriOptions);
        } catch (e) {
            logger.error('[Notification] Tauri sendNotification error:', e);
            return null;
        }
    }

    if (env.isWeb() && typeof window !== 'undefined' && 'Notification' in window) {
        try {
            const webTag = options.tag || `tb-notif-${Date.now()}-${Math.random().toString(36).substring(2,7)}`;
            const webOptions = {
                body: options.body,
                icon: options.icon,
                tag: webTag,
                data: options.data || { tbjsData: { ...options }, originalTag: options.tag, internalTag: webTag },
                silent: options.silent === undefined ? (options.sound ? false : true) : options.silent, // if sound string implies not silent.
                renotify: options.renotify || false,
                requireInteraction: options.requireInteraction || false,
                actions: options.actions ? options.actions.map(a => ({ action: a.id, title: a.title, icon: a.icon })) : undefined,
            };

            const notification = new Notification(options.title, webOptions);
            activeWebNotifications.set(webTag, { notification, options });

            notification.onclick = (event) => {
                logger.log('[Notification] Web notification clicked:', event, webTag);
                const payload = {
                    actionId: event.action || 'default', // event.action is available if SW handles 'notificationclick'
                    notificationId: webTag,
                    data: webOptions.data?.tbjsData || webOptions.data, // Use original data
                };
                eventBus.emit('tbjs:notification:action', payload);
                window.focus(); // Bring window to front on click
            };
            notification.onshow = () => logger.log('[Notification] Web notification shown:', webTag);
            notification.onerror = (err) => {
                logger.error('[Notification] Web notification error:', err, webTag);
                activeWebNotifications.delete(webTag);
            };
            notification.onclose = () => {
                logger.log('[Notification] Web notification closed:', webTag);
                activeWebNotifications.delete(webTag);
            };
            return notification;
        } catch (e) {
            logger.error('[Notification] Web sendNotification error:', e);
            return null;
        }
    }
    logger.warn('[Notification] Platform not supported for sendNotification.');
    return null;
}

// --- Actions ---
/** @typedef {import('@tauri-apps/plugin-notification').ActionTypeDefinition} TauriActionTypeDefinition */

/**
 * Registers action types. (Primarily for Tauri Mobile)
 * @param {Array<TauriActionTypeDefinition>} actionTypes
 * @returns {Promise<void>}
 */
async function registerActionTypes(actionTypes) {
    if (env.isTauri() && tauriRegisterActionTypes) {
        try {
            return await tauriRegisterActionTypes(actionTypes);
        } catch (e) {
            logger.error('[Notification] Tauri registerActionTypes error:', e);
        }
    } else if (env.isWeb()) {
        logger.warn('[Notification] Web: `registerActionTypes` is a no-op. Actions are defined per-notification.');
    } else {
        logger.warn('[Notification] Platform not supported for registerActionTypes.');
    }
}

/** @typedef {import('@tauri-apps/plugin-notification').NotificationActionPerformed} TauriNotificationActionPerformed */
/**
 * @typedef {Object} TBNotificationActionPayload
 * @property {string} actionId - The ID of the action performed.
 * @property {string | number} notificationId - The ID (Tauri numeric id or Web tag) of the notification.
 * @property {string} [inputValue] - (Tauri Mobile) Value from text input action.
 * @property {any} [data] - Custom data from the original notification. (Web: from `data.tbjsData`, Tauri: from `notification.extra`)
 */

/**
 * Listens for notification actions.
 * @param {(payload: TBNotificationActionPayload) => void} callback
 * @returns {Promise<() => void> | (() => void)} Unsubscribe function.
 */
function onAction(callback) {
    if (env.isTauri() && tauriOnAction) {
        try {
            return tauriOnAction((event /*: TauriNotificationActionPerformed */) => {
                const payload = {
                    actionId: event.actionId,
                    notificationId: event.id,
                    inputValue: event.inputValue,
                    data: event.notification?.extra,
                };
                callback(payload);
            });
        } catch (e) {
            logger.error('[Notification] Tauri onAction error:', e);
            return () => {};
        }
    } else if (env.isWeb()) {
        const handler = (payload) => callback(payload);
        eventBus.on('tbjs:notification:action', handler);
        logger.info('[Notification] Web onAction registered. Relies on internal event bus.');
        return () => eventBus.off('tbjs:notification:action', handler);
    }
    logger.warn('[Notification] Platform not supported for onAction.');
    return () => {};
}

// --- Channels (Primarily Tauri Android) ---
/** @typedef {import('@tauri-apps/plugin-notification').Channel} TauriChannel */

/**
 * Creates a notification channel.
 * @param {TauriChannel} channel
 * @returns {Promise<void>}
 */
async function createChannel(channel) {
    if (env.isTauri() && tauriCreateChannel) {
        try {
            return await tauriCreateChannel(channel);
        } catch (e) {
            logger.error('[Notification] Tauri createChannel error:', e);
        }
    } else {
        logger.warn('[Notification] Channels are not supported on this platform or plugin unavailable.');
    }
}

/**
 * Deletes a notification channel.
 * @param {string} channelId
 * @returns {Promise<void>}
 */
async function deleteChannel(channelId) {
    if (env.isTauri() && tauriDeleteChannel) {
        try {
            return await tauriDeleteChannel(channelId);
        } catch (e) {
            logger.error('[Notification] Tauri deleteChannel error:', e);
        }
    } else {
        logger.warn('[Notification] Channels are not supported on this platform or plugin unavailable.');
    }
}

/**
 * Lists all notification channels.
 * @returns {Promise<Array<TauriChannel>>}
 */
async function listChannels() {
    if (env.isTauri() && tauriListChannels) {
        try {
            return await tauriListChannels();
        } catch (e) {
            logger.error('[Notification] Tauri listChannels error:', e);
            return [];
        }
    }
    logger.warn('[Notification] Channels are not supported on this platform or plugin unavailable.');
    return [];
}

// --- Management ---
/**
 * Cancels/closes a specific notification.
 * For Tauri, `idOrTag` can be a numeric ID (for scheduled or active) or a string tag (for active).
 * For Web, `idOrTag` is the notification tag (string).
 * @param {string | number} idOrTag - The ID or tag of the notification.
 * @returns {Promise<void>}
 */
async function cancel(idOrTag) {
    if (env.isTauri()) {
        try {
            if (typeof idOrTag === 'number') {
                // Try to cancel as scheduled first, then as active if removeActive is available
                if (tauriCancelScheduled) {
                    try {
                        await tauriCancelScheduled(idOrTag); // For scheduled
                        return;
                    } catch (scheduledError) {
                        // Might not be a scheduled notification, or error occurred. Try removeActive.
                        if (tauriRemoveActive) {
                            await tauriRemoveActive([{ id: idOrTag }]); // For active by numeric ID
                            return;
                        }
                        throw scheduledError; // Re-throw if removeActive not an option
                    }
                } else if (tauriRemoveActive) {
                    await tauriRemoveActive([{ id: idOrTag }]); // For active by numeric ID
                    return;
                }
            } else if (typeof idOrTag === 'string' && tauriRemoveActive) {
                await tauriRemoveActive([{ tag: idOrTag }]); // For active by tag
                return;
            }
            logger.warn('[Notification] Tauri: No suitable cancel/removeActive function or idOrTag type for:', idOrTag);
        } catch (e) {
            logger.error('[Notification] Tauri cancel/removeActive notification error:', e);
        }
    } else if (env.isWeb()) {
        const entry = activeWebNotifications.get(String(idOrTag));
        if (entry && entry.notification) {
            entry.notification.close();
            // activeWebNotifications.delete(String(idOrTag)); // onclose handles this
        } else {
            logger.warn('[Notification] Web: Notification to cancel not found by tag:', idOrTag);
        }
    } else {
        logger.warn('[Notification] Platform not supported for cancel.');
    }
}

/**
 * Cancels/closes all notifications.
 * @returns {Promise<void>}
 */
async function cancelAll() {
    if (env.isTauri()) {
        if (tauriRemoveActive && tauriGetActive) {
            try {
                const activeNotifications = await tauriGetActive();
                if (activeNotifications && activeNotifications.length > 0) {
                    const idsToRemove = activeNotifications.map(n => ({ id: n.id })); // Needs numeric ID
                    await tauriRemoveActive(idsToRemove);
                }
                // How to cancel all scheduled? Tauri docs don't show a clear "cancelAllScheduled".
                // May need to getPending then cancel one by one.
            } catch (e) {
                logger.error('[Notification] Tauri cancelAll (active) error:', e);
            }
        } else {
            logger.warn('[Notification] Tauri: removeActive or getActive not available for cancelAll.');
        }
    } else if (env.isWeb()) {
        activeWebNotifications.forEach(entry => {
            if (entry.notification) entry.notification.close();
        });
        // activeWebNotifications.clear(); // onclose handles this
    } else {
        logger.warn('[Notification] Platform not supported for cancelAll.');
    }
}

/** @typedef {import('@tauri-apps/plugin-notification').ActiveNotification} TauriActiveNotification */
/**
 * Gets currently active/displayed notifications.
 * @returns {Promise<Array<TauriActiveNotification | NotificationOptions>>} Array of notifications.
 */
async function getActive() {
    if (env.isTauri() && tauriGetActive) {
        try {
            return await tauriGetActive();
        } catch (e) {
            logger.error('[Notification] Tauri getActive error:', e);
            return [];
        }
    } else if (env.isWeb()) {
        return Array.from(activeWebNotifications.values()).map(entry => entry.options);
    }
    logger.warn('[Notification] Platform not supported for getActive.');
    return [];
}

/** @typedef {import('@tauri-apps/plugin-notification').PendingNotification} TauriPendingNotification */
/**
 * Gets pending/scheduled notifications. (Mainly Tauri)
 * @returns {Promise<Array<TauriPendingNotification>>}
 */
async function getPending() {
    if (env.isTauri() && tauriGetPending) {
        try {
            return await tauriGetPending();
        } catch (e) {
            logger.error('[Notification] Tauri getPending error:', e);
            return [];
        }
    }
    logger.warn('[Notification] getPending is primarily a Tauri feature and not supported on Web.');
    return [];
}

const notificationModule = {
    isPermissionGranted,
    requestPermission,
    checkPermissions,
    sendNotification,
    registerActionTypes,
    onAction,
    createChannel,
    deleteChannel,
    listChannels,
    cancel,
    cancelAll,
    getActive,
    getPending,
    Importance,
    Visibility,
};

export default notificationModule;
