import { Navigate, Outlet } from "react-router-dom";
import { getToken } from "@/lib/api";

export default function RequireAuth() {
  if (getToken() === null) return <Navigate to="/login" replace />;
  return <Outlet />;
}
