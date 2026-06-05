// lib/services/auth_service.dart

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';
import 'package:google_sign_in/google_sign_in.dart';

import '../models/models.dart';

class AuthService {
  final _auth = FirebaseAuth.instance;

  // Web requires clientId explicitly — mobile gets it from google-services.json
  GoogleSignIn get _google => GoogleSignIn(
        clientId: kIsWeb
            ? '807290065452-0fsihbaj7vnohgqslkpogurf67ah7f5l.apps.googleusercontent.com'
            : null,
      );

  // ── Current user stream ───────────────────────────────────────────────────

  Stream<ResumeUser?> get userStream => _auth.authStateChanges().map((u) {
        if (u == null) return null;
        return ResumeUser.fromFirebase(
          uid: u.uid,
          displayName: u.displayName,
          email: u.email,
          photoUrl: u.photoURL,
        );
      });

  ResumeUser? get currentUser {
    final u = _auth.currentUser;
    if (u == null) return null;
    return ResumeUser.fromFirebase(
      uid: u.uid,
      displayName: u.displayName,
      email: u.email,
      photoUrl: u.photoURL,
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
    await cred.user!.updateDisplayName(name.trim());
    await cred.user!.reload();
    final updated = _auth.currentUser!;
    return ResumeUser.fromFirebase(
      uid: updated.uid,
      displayName: updated.displayName,
      email: updated.email,
      photoUrl: updated.photoURL,
    );
  }

  Future<void> sendPasswordReset(String email) async {
    await _auth.sendPasswordResetEmail(email: email.trim());
  }

  // ── Google Sign-In ────────────────────────────────────────────────────────

  Future<ResumeUser> signInWithGoogle() async {
    if (kIsWeb) {
      // Web uses popup flow — no GoogleSignIn package needed
      final provider = GoogleAuthProvider();
      provider.addScope('email');
      provider.addScope('profile');
      final cred = await _auth.signInWithPopup(provider);
      return _fromCred(cred);
    }

    // Mobile flow
    final googleUser = await _google.signIn();
    if (googleUser == null) throw Exception('Google sign-in cancelled.');

    final googleAuth = await googleUser.authentication;
    final credential = GoogleAuthProvider.credential(
      accessToken: googleAuth.accessToken,
      idToken: googleAuth.idToken,
    );

    final cred = await _auth.signInWithCredential(credential);
    return _fromCred(cred);
  }

  // ── Guest ─────────────────────────────────────────────────────────────────

  ResumeUser continueAsGuest() => ResumeUser.guest();

  // ── Sign out ──────────────────────────────────────────────────────────────

  Future<void> signOut() async {
    await _auth.signOut();
    if (!kIsWeb) await _google.signOut();
  }

  // ── Helper ────────────────────────────────────────────────────────────────

  ResumeUser _fromCred(UserCredential cred) {
    final u = cred.user!;
    return ResumeUser.fromFirebase(
      uid: u.uid,
      displayName: u.displayName,
      email: u.email,
      photoUrl: u.photoURL,
    );
  }
}
