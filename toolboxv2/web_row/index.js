navigator.serviceWorker.getRegistrations().then(function(registrations) {
    registrations.forEach(function(registration) {
        registration.unregister().then(function(success) {
            if (success) {
                console.log('ServiceWorker mit dem Namen', registration.scope, 'wurde erfolgreich entfernt.');
            } else {
                console.log('Fehler beim Entfernen des ServiceWorker mit dem Namen', registration.scope);
            }
        });
    });
});

/*
const registerServiceWorker = async () => {
  if ("serviceWorker" in navigator) {
    try {
      const registration = await navigator.serviceWorker.register("/web/sw.js", {
        scope: "/web/",
      });
      if (registration.installing) {
        console.log("Service worker installing");
      } else if (registration.waiting) {
        console.log("Service worker installed");
      } else if (registration.active) {
        console.log("Service worker active");
      }
    } catch (error) {
      console.error(`Registration failed with ${error}`);
    }
  }else {
  console.log("Service worker not found")}
};

registerServiceWorker();
*/

