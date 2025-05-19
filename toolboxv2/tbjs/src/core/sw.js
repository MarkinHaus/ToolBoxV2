// tbjs/src/core/sw.js (Service Worker Manager Module)
import logger from './logger.js';
import config from './config.js';
import events from './events.js';

const ServiceWorkerManager = {
    SW_SCRIPT_URL: '/sw.js', // Default path to the actual service worker file

    async register() {
        const swConfig = config.get('serviceWorker');
        if (!swConfig || !swConfig.enabled) {
            logger.info('[SW] Service Worker registration is disabled by config.');
            return null;
        }

        if (!('serviceWorker' in navigator)) {
            logger.warn('[SW] Service Worker not supported in this browser.');
            return null;
        }

        const swUrl = swConfig.url || this.SW_SCRIPT_URL;
        const swScope = swConfig.scope || '/';

        try {
            const registration = await navigator.serviceWorker.register(swUrl, {
                scope: swScope,
            });
            logger.log(`[SW] Service Worker registered with URL ${swUrl} and scope: ${registration.scope}`);

            registration.onupdatefound = () => {
                const installingWorker = registration.installing;
                if (installingWorker) {
                    logger.log('[SW] Update found. New service worker installing.');
                    installingWorker.onstatechange = () => {
                        if (installingWorker.state === 'installed') {
                            if (navigator.serviceWorker.controller) {
                                logger.log('[SW] New content is available and will be used when all tabs for this page are closed.');
                                events.emit('sw:updateAvailable', { registration });
                            } else {
                                logger.log('[SW] Content is cached for offline use.');
                                events.emit('sw:contentCached', { registration });
                            }
                        } else if (installingWorker.state === 'redundant') {
                            logger.warn('[SW] Installing service worker became redundant.');
                        }
                    };
                }
            };
            return registration;
        } catch (error) {
            logger.error(`[SW] Service Worker registration failed for URL ${swUrl} with scope ${swScope}:`, error);
            return null;
        }
    },

    async unregister() {
        if (!('serviceWorker' in navigator)) {
            logger.warn('[SW] Service Worker not supported.');
            return false;
        }
        try {
            const registrations = await navigator.serviceWorker.getRegistrations();
            let unregistered = false;
            for (const registration of registrations) {
                await registration.unregister();
                logger.log(`[SW] Service Worker unregistered for scope: ${registration.scope}`);
                unregistered = true;
            }
            if (!unregistered) logger.log('[SW] No active service workers found to unregister.');
            return true;
        } catch (error) {
            logger.error('[SW] Service Worker unregistration failed:', error);
            return false;
        }
    },

    async sendMessage(message) {
        if (!navigator.serviceWorker.controller) {
            logger.warn('[SW] No active service worker controller to send message to.');
            return Promise.reject('No active service worker controller.');
        }
        return new Promise((resolve, reject) => {
            const messageChannel = new MessageChannel();
            messageChannel.port1.onmessage = (event) => {
                if (event.data && event.data.error) {
                    logger.error('[SW] Message response error:', event.data.error);
                    reject(event.data.error);
                } else {
                    resolve(event.data);
                }
            };
            navigator.serviceWorker.controller.postMessage(message, [messageChannel.port2]);
        });
    }
};

export default ServiceWorkerManager;
