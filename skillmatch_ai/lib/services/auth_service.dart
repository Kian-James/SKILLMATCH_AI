// services/auth_service.dart
// Wraps Firebase Auth for Email/Password, Google Sign-In, and Anonymous login.
// Persists sessions automatically via Firebase's built-in token refresh.

import 'package:flutter/foundation.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';

class AuthService extends ChangeNotifier {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final GoogleSignIn _googleSignIn = GoogleSignIn();

  /// Stream of auth state changes — used by AuthGate to route users
  Stream<User?> get authStateChanges => _auth.authStateChanges();

  /// Currently authenticated user (null if not logged in)
  User? get currentUser => _auth.currentUser;

  /// True if current user signed in anonymously
  bool get isAnonymous => currentUser?.isAnonymous ?? false;

  /// Firebase ID token — sent as Bearer token to FastAPI backend
  Future<String?> getIdToken({bool forceRefresh = false}) async {
    return await currentUser?.getIdToken(forceRefresh);
  }

  // ─── Email & Password ──────────────────────────────────────────────────────

  Future<UserCredential> signInWithEmail(String email, String password) async {
    return await _auth.signInWithEmailAndPassword(
      email: email.trim(),
      password: password,
    );
  }

  Future<UserCredential> registerWithEmail(
    String email,
    String password,
    String displayName,
  ) async {
    final cred = await _auth.createUserWithEmailAndPassword(
      email: email.trim(),
      password: password,
    );
    await cred.user?.updateDisplayName(displayName.trim());
    return cred;
  }

  Future<void> sendPasswordReset(String email) async {
    await _auth.sendPasswordResetEmail(email: email.trim());
  }

  // ─── Google Sign-In ────────────────────────────────────────────────────────

  Future<UserCredential?> signInWithGoogle() async {
    final googleUser = await _googleSignIn.signIn();
    if (googleUser == null) return null; // User cancelled

    final googleAuth = await googleUser.authentication;
    final credential = GoogleAuthProvider.credential(
      accessToken: googleAuth.accessToken,
      idToken: googleAuth.idToken,
    );

    // If user was anonymous, link Google account to preserve their data
    if (isAnonymous && currentUser != null) {
      try {
        return await currentUser!.linkWithCredential(credential);
      } on FirebaseAuthException catch (e) {
        if (e.code == 'credential-already-in-use') {
          // Account already exists — sign in normally
          return await _auth.signInWithCredential(credential);
        }
        rethrow;
      }
    }

    return await _auth.signInWithCredential(credential);
  }

  // ─── Anonymous ─────────────────────────────────────────────────────────────

  Future<UserCredential> signInAnonymously() async {
    return await _auth.signInAnonymously();
  }

  /// Upgrade anonymous user to email/password (preserves all their data)
  Future<UserCredential> linkAnonymousWithEmail(
    String email,
    String password,
    String displayName,
  ) async {
    final credential = EmailAuthProvider.credential(
      email: email.trim(),
      password: password,
    );
    final linked = await currentUser!.linkWithCredential(credential);
    await linked.user?.updateDisplayName(displayName.trim());
    return linked;
  }

  // ─── Sign Out ──────────────────────────────────────────────────────────────

  Future<void> signOut() async {
    await _googleSignIn.signOut();
    await _auth.signOut();
  }
}
