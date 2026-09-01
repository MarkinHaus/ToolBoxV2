/* Service Worker — cache-first, Version bump beim Deploy. */
var CACHE = 'habits-v3';
var ASSETS = ['./', './index.html', './style.css', './app.js', './ui.js', './manifest.json', './icons/icon-192.png', './icons/icon-512.png'];

self.addEventListener('install', function (e) {
  e.waitUntil(caches.open(CACHE).then(function (c) { return c.addAll(ASSETS); }).then(function () { return self.skipWaiting(); }));
});
self.addEventListener('activate', function (e) {
  e.waitUntil(caches.keys().then(function (keys) {
    return Promise.all(keys.filter(function (k) { return k !== CACHE; }).map(function (k) { return caches.delete(k); }));
  }).then(function () { return self.clients.claim(); }));
});
self.addEventListener('fetch', function (e) {
  if (e.request.method !== 'GET') return;
  e.respondWith(
    caches.match(e.request).then(function (hit) {
      var fetchUp = fetch(e.request).then(function (res) {
        if (res && res.ok) { var cp = res.clone(); caches.open(CACHE).then(function (c) { c.put(e.request, cp); }); }
        return res;
      }).catch(function () { return hit; });
      return hit || fetchUp;
    })
  );
});
