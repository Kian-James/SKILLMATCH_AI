import 'package:firebase_auth/firebase_auth.dart';
import '../config/firebase_config.dart';

class AuthService {
  final FirebaseAuth _auth = FirebaseConfig.auth;

  User? get currentUser => _auth.currentUser;

  Future<User?> signInWithEmail(String email, String password) async {
    final cred = await _auth.signInWithEmailAndPassword(
        email: email, password: password);
    return cred.user;
  }

  Future<User?> signUpWithEmail(String email, String password) async {
    final cred = await _auth.createUserWithEmailAndPassword(
        email: email, password: password);
    return cred.user;
  }

  Future<void> signOut() async {
    await _auth.signOut();
  }
}
