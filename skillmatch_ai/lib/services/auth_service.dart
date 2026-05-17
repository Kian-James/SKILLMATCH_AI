// lib/services/auth_service.dart

import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';
import '../models/models.dart';

class AuthService {
  final _auth   = FirebaseAuth.instance;
  final _google = GoogleSignIn();

  // ── Current user stream ───────────────────────────────────────────────────

  /// Emits a [ResumeUser] whenever auth state changes, null when signed out.
  Stream<ResumeUser?> get userStream => _auth.authStateChanges().map((u) {
        if (u == null) return null;
        return ResumeUser.fromFirebase(
          uid:         u.uid,
          displayName: u.displayName,
          email:       u.email,
          photoUrl:    u.photoURL,
        );
      });

  ResumeUser? get currentUser {
    final u = _auth.currentUser;
    if (u == null) return null;
    return ResumeUser.fromFirebase(
      uid:         u.uid,
      displayName: u.displayName,
      email:       u.email,
      photoUrl:    u.photoURL,
    );
  }

  // ── Email / password ──────────────────────────────────────────────────────

  Future<ResumeUser> signInWithEmail({
    required String email,
    required String password,
  }) async {
    final cred = await _auth.signInWithEmailAndPassword(
        email: email.trim(), password: password);
    return _fromCred(cred);
  }

  Future<ResumeUser> signUpWithEmail({
    required String name,
    required String email,
    required String password,
  }) async {
    final cred = await _auth.createUserWithEmailAndPassword(
        email: email.trim(), password: password);
    // Set display name immediately after creation
    await cred.user!.updateDisplayName(name.trim());
    await cred.user!.reload();
    final updated = _auth.currentUser!;
    return ResumeUser.fromFirebase(
      uid:         updated.uid,
      displayName: updated.displayName,
      email:       updated.email,
      photoUrl:    updated.photoURL,
    );
  }

  Future<void> sendPasswordReset(String email) async {
    await _auth.sendPasswordResetEmail(email: email.trim());
  }

  // ── Google Sign-In ────────────────────────────────────────────────────────

  Future<ResumeUser> signInWithGoogle() async {
    final googleUser = await _google.signIn();
    if (googleUser == null) throw Exception('Google sign-in cancelled.');

    final googleAuth = await googleUser.authentication;
    final credential = GoogleAuthProvider.credential(
      accessToken: googleAuth.accessToken,
      idToken:     googleAuth.idToken,
    );

    final cred = await _auth.signInWithCredential(credential);
    return _fromCred(cred);
  }

  // ── Guest ─────────────────────────────────────────────────────────────────

  /// Returns a local-only guest user — no Firebase account created.
  /// Guest history is stored under their temporary ID and is lost on sign-out.
  ResumeUser continueAsGuest() => ResumeUser.guest();

  // ── Sign out ──────────────────────────────────────────────────────────────

  Future<void> signOut() async {
    await Future.wait([
      _auth.signOut(),
      _google.signOut(),
    ]);
  }

  // ── Helper ────────────────────────────────────────────────────────────────

  ResumeUser _fromCred(UserCredential cred) {
    final u = cred.user!;
    return ResumeUser.fromFirebase(
      uid:         u.uid,
      displayName: u.displayName,
      email:       u.email,
      photoUrl:    u.photoURL,
    );
  }
}
