import { css } from '@emotion/css';
import { GrafanaTheme2 } from '@grafana/data';

/**
 * Shared loading state container
 */
export const getLoadingContainerStyles = (theme: GrafanaTheme2) => css`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: ${theme.colors.text.secondary};
`;

/**
 * Shared error state container
 */
export const getErrorContainerStyles = (theme: GrafanaTheme2) => css`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: ${theme.spacing(2)};
`;
