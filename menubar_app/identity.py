"""
menubar_app/identity.py
========================
Cross-platform contact identity resolver.

Maps a contact name (from Messages/WA Desktop) or email address (from Gmail)
to a stable canonical entity_id used as conversation_id across all platforms.

This is the Phase 2 cross-platform link:
  Messages   "John Smith" → Contacts → john@company.com → entity_a3f9b2c1
  Gmail      "john@company.com"       → entity_a3f9b2c1   ← SAME WINDOW
  WA Web     "John Smith"             → entity_a3f9b2c1   ← SAME WINDOW

Requires permission: System Preferences → Privacy & Security → Contacts

Falls back gracefully at every level:
  Contacts lookup fails → hash the name/email directly
  No name/email        → platform-specific ID (isolated window, no cross-link)
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ── Permission check ──────────────────────────────────────────────────────────

def check_contacts_permission() -> bool:
    """
    Returns True if Contacts access has been granted.
    CNContactStore.authorizationStatusForEntityType_ returns:
      0 = NotDetermined, 1 = Restricted, 2 = Denied, 3 = Authorized
    """
    try:
        from Contacts import CNContactStore
        # CNEntityTypeContacts = 0 in pyobjc (integer constant, not class attr)
        return CNContactStore.authorizationStatusForEntityType_(0) == 3
    except Exception:
        return False


def request_contacts_permission() -> None:
    """Request Contacts permission — triggers system dialog on first call."""
    try:
        from Contacts import CNContactStore
        store = CNContactStore.alloc().init()
        # 0 = CNEntityTypeContacts
        store.requestAccessForEntityType_completionHandler_(
            0,
            lambda granted, err: logger.info("Contacts permission: %s", granted),
        )
    except Exception as e:
        logger.error("Could not request Contacts permission: %s", e)


# ── Public interface ──────────────────────────────────────────────────────────

def resolve(
    name:  Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
) -> str:
    """
    Resolve a contact to a stable entity_id string.

    Priority:
      1. email provided directly → use as canonical (already unique)
      2. name provided → look up in Contacts → find email → use as canonical
      3. phone provided → look up in Contacts → find email → use as canonical
      4. Fallback → hash whatever we have

    Returns "entity_<12-char hex>" — stable across platforms for same person.
    """
    # Email is already a canonical cross-platform identifier
    if email and "@" in email:
        return _make_id(email.lower().strip())

    # Try Contacts lookup
    if name or phone:
        canonical = _lookup_canonical(name=name, phone=phone)
        if canonical:
            logger.debug("Resolved '%s' → '%s'", name or phone, canonical)
            return _make_id(canonical)

    # Fallback: hash what we have
    fallback = (name or phone or "unknown").strip().lower()
    return _make_id(fallback)


# ── Contacts lookup ───────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def _lookup_canonical(
    name:  Optional[str] = None,
    phone: Optional[str] = None,
) -> Optional[str]:
    """
    Query macOS Contacts (CNContactStore) and return the primary email
    for the contact matching name or phone number.

    Results are cached — same name always resolves to the same email
    within a session without re-querying Contacts.
    """
    try:
        from Contacts import (
            CNContactStore,
            CNContact,
            CNContactEmailAddressesKey,
            CNContactGivenNameKey,
            CNContactFamilyNameKey,
            CNContactPhoneNumbersKey,
        )

        # Check permission — 0 = CNEntityTypeContacts, 3 = Authorized
        store = CNContactStore.alloc().init()
        if CNContactStore.authorizationStatusForEntityType_(0) != 3:
            logger.debug("Contacts permission not granted")
            return None

        keys = [
            CNContactEmailAddressesKey,
            CNContactGivenNameKey,
            CNContactFamilyNameKey,
            CNContactPhoneNumbersKey,
        ]

        contacts = []

        # Search by name
        if name:
            predicate = CNContact.predicateForContactsMatchingName_(name.strip())
            err_ref = None
            result = store.unifiedContactsMatchingPredicate_keysToFetch_error_(
                predicate, keys, None
            )
            contacts.extend(result or [])

        # Search by phone number
        if phone and not contacts:
            from Contacts import CNPhoneNumber
            predicate = CNContact.predicateForContactsMatchingPhoneNumber_(
                CNPhoneNumber.phoneNumberWithStringValue_(phone)
            )
            result = store.unifiedContactsMatchingPredicate_keysToFetch_error_(
                predicate, keys, None
            )
            contacts.extend(result or [])

        # Return first email found
        for contact in contacts:
            emails = contact.emailAddresses()
            if emails and len(emails) > 0:
                email_val = str(emails[0].value())
                if email_val and "@" in email_val:
                    return email_val.lower().strip()

        # Contact found but no email — use full name as canonical
        if contacts:
            c = contacts[0]
            given  = str(c.givenName()  or "").strip()
            family = str(c.familyName() or "").strip()
            full   = f"{given} {family}".strip()
            if full:
                return full.lower()

    except Exception as e:
        logger.debug("Contacts lookup error: %s", e)

    return None


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_id(canonical: str) -> str:
    """Produce a stable 12-char hex entity_id from a canonical string."""
    return "entity_" + hashlib.md5(canonical.encode()).hexdigest()[:12]