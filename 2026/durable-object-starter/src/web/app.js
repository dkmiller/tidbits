/*
 * Browser half of the WebAuthn ceremonies, built on @simplewebauthn/browser.
 * That library owns the ArrayBuffer <-> base64url encoding on both sides of
 * navigator.credentials, so this file is just the two HTTP round trips.
 *
 * Loaded as an ES module (see index.html), so the import below needs no
 * bundler — the browser fetches it directly from the CDN.
 */

import {
  startRegistration,
  browserSupportsWebAuthn,
} from "https://cdn.jsdelivr.net/npm/@simplewebauthn/browser@10/+esm";

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    body: JSON.stringify(body),
  });

  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(payload.detail || "The server rejected that request.");
  }
  return payload;
}

/** Turn the browser's terse DOMExceptions into something a person can act on. */
function describeError(err) {
  switch (err.name) {
    case "NotAllowedError":
      return "The request was dismissed or timed out. Try again.";
    case "InvalidStateError":
      return "This device already has a passkey for that account.";
    case "NotSupportedError":
      return "This device can't create the kind of passkey this site needs.";
    case "SecurityError":
      return "The page origin doesn't match the site's passkey settings.";
    default:
      return err.message || "Something went wrong. Try again.";
  }
}

async function registerVisit(username) {
  // /auth/register/begin returns { challengeId, options } — options is the
  // PublicKeyCredentialCreationOptionsJSON shape startRegistration expects.
  const { challengeId, options } = await postJSON("/auth/register/begin", {
    username,
    displayName: username,
  });

  const credential = await startRegistration({ optionsJSON: options });

  return postJSON("/auth/register/finish", { challengeId, credential });
}

document.addEventListener("DOMContentLoaded", () => {
  const usernameInput = document.querySelector("#username");
  const registerButton = document.querySelector("#register");
  const status = document.querySelector("#status");

  const show = (message) => {
    status.textContent = message;
  };

  if (!browserSupportsWebAuthn()) {
    show("This browser doesn't support passkeys.");
    registerButton.disabled = true;
    return;
  }

  registerButton.addEventListener("click", async () => {
    const username = usernameInput.value.trim();
    if (!username) {
      show("Enter a username first.");
      return;
    }

    registerButton.disabled = true;
    show("Follow your browser's prompt to create a passkey…");

    try {
      await registerVisit(username);
      show(`Passkey created for ${username}. You're signed in.`);
    } catch (err) {
      show(describeError(err));
    } finally {
      registerButton.disabled = false;
    }
  });
});
